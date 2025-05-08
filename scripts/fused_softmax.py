# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
import math
from megatron.core.fusions.ScalableSoftmax import ScalableSoftmax

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import get_default_causal_mask


class ScaleMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, scale, s, use_ssmax, mask_func, attn_mask_type, input_in_float16, softmax_in_fp32):
        if input_in_float16 and softmax_in_fp32:
            input = input.float()

        if scale is not None:
            input = input * scale

        # Generate causal mask if not given
        sq, sk = input.size(2), input.size(3)
        if attn_mask_type == AttnMaskType.causal and mask is None and sq > 1:
            assert sq == sk, "causal mask is only for self attention"
            mask = get_default_causal_mask(sq)

        mask_output = mask_func(input, mask) if mask is not None else input
        
       
        log_seq_len = torch.log(torch.tensor(sq, device=input.device))
        s = s * log_seq_len
        if use_ssmax:
            mask_output = mask_output * s.unsqueeze(-1).unsqueeze(-1)
            
        probs = torch.nn.Softmax(-1)(mask_output)
        ctx.save_for_backward(mask_output, log_seq_len, probs)
        if input_in_float16 and softmax_in_fp32:
            probs = probs.half() if input.dtype == torch.float16 else probs.bfloat16()

        return probs

    @staticmethod
    def backward(ctx, grad_output):
        mask_output, log_seq_len, softmax_output= ctx.saved_tensors
        # 计算sum(softmax_output * mask_output)
        sum_softmax_input = torch.sum(softmax_output * mask_output, dim=-1, keepdim=True)
        
        # 计算梯度分量
        grad_component = mask_output - sum_softmax_input
        
        # 计算最终的s梯度
        grad_s = torch.sum(grad_output * softmax_output * grad_component) * log_seq_len
        # 需要把前置attn score probs 传入, lambda
        lambda = 0.9
        expert_balance_loss = torch.var(probs, dim=-1).mean()
        grad_s = grad_s + lambda * expert_balance_loss
        
        
        return (
            None,    # input梯度
            None,    # mask梯度
            None,    # scale梯度 (不需要)
            grad_s,  # s梯度
            None,    # use_ssmax梯度
            None,    # mask_func梯度
            None,    # attn_mask_type梯度
            None,    # input_in_float16梯度
            None     # softmax_in_fp32梯度
        )

class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale.item())
        ctx.save_for_backward(inputs, softmax_results, scale)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda
        inputs, softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t.item()
        )
        scale_grads = torch.sum(output_grads * softmax_results * inputs)

        return input_grads, scale_grads.view_as(scale_t)

class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda
        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale.item())
        ctx.save_for_backward(inputs, softmax_results, scale)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda
        inputs, softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t.item())
        scale_grads = torch.sum(output_grads * softmax_results * inputs)

        return input_grads, None, scale_grads.view_as(scale_t)

class ScaledSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_softmax_cuda
        softmax_results = scaled_softmax_cuda.forward(inputs, scale.item())
        ctx.save_for_backward(inputs, softmax_results, scale)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_softmax_cuda
        inputs, softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_softmax_cuda.backward(output_grads, softmax_results, scale_t.item())
        scale_grads = torch.sum(output_grads * softmax_results * inputs)
    
        return input_grads, scale_grads.view_as(scale_t)

class FusedScaleMaskSoftmax(nn.Module):
    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
        use_ssmax
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (self.input_in_fp16 and self.input_in_bf16)
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.use_ssmax = use_ssmax

        if self.use_ssmax:
            self.s = nn.Parameter(torch.tensor(0.43, requires_grad=True))
        else:
            self.register_buffer("s", torch.tensor(0.43))

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]):
        assert input.dim() == 4

        if self.is_kernel_available(mask, *input.size()):
            
            return self.forward_fused_softmax(input, mask)
        else:
            
     
            return ScaleMaskSoftmaxFunction.apply(
                input, mask, self.scale, self.s, self.use_ssmax,
                self.mask_func, self.attn_mask_type, 
                self.input_in_float16, self.softmax_in_fp32
            )

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if (
            self.scaled_masked_softmax_fusion
            and self.input_in_float16
            and 16 < sk <= 4096
            and sq % 4 == 0
            and sk % 4 == 0
            and attn_batches % 4 == 0
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.attn_mask_type == AttnMaskType.causal:
                    return attn_batches % batch_per_block == 0
                else:
                    return sq % batch_per_block == 0
        return False

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        if self.use_ssmax:
            s = self.s * math.log(torch.tensor(sq, device=input.device))
        else:
            s = self.scale if self.scale is not None else 1.0
            s = torch.tensor([s])
        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk
            input = input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, s)
            return probs.view(b, np, sq, sk)
        else:
            if mask is not None:
                return ScaledMaskedSoftmax.apply(input, mask, s)
            else:
                return ScaledSoftmax.apply(input, s)

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda
        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
