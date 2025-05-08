# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from verl.trainer.ppo.core_algos import compute_difficulty_coeff, generate_curriculum
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import pandas as pd
from verl import DataProto
#from verl.utils.text.tokenizer import load_tokenizer

def collate_fn3(batch: List[Dict]) -> Dict:
    """
    Collate function for RLHFDataset, ensuring compatibility with DAPO.

    Args:
        batch: List of dictionaries from __getitem__.

    Returns:
        Dictionary with batched tensors and non-tensor data.
    """
    # Initialize lists for batch data
    input_ids = []
    attention_mask = []
    non_tensor_batch = {
        "prompt": [],
        "data_source": [],
    }
    has_pairs = "positive_response" in batch[0]
    if has_pairs:
        positive_response_ids = []
        positive_response_mask = []
        negative_response_ids = []
        negative_response_mask = []
        non_tensor_batch["positive_response"] = []
        non_tensor_batch["negative_response"] = []

    # Collect data
    for item in batch:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        non_tensor_batch["prompt"].append(item["prompt"])
        non_tensor_batch["data_source"].append(item["data_source"])
        if has_pairs:
            positive_response_ids.append(item["positive_response_ids"])
            positive_response_mask.append(item["positive_response_mask"])
            negative_response_ids.append(item["negative_response_ids"])
            negative_response_mask.append(item["negative_response_mask"])
            non_tensor_batch["positive_response"].append(item["positive_response"])
            non_tensor_batch["negative_response"].append(item["negative_response"])
        if "raw_prompt_ids" in item:
            non_tensor_batch.setdefault("raw_prompt_ids", []).append(item["raw_prompt_ids"])
        if "raw_chat" in item:
            non_tensor_batch.setdefault("raw_chat", []).append(item["raw_chat"])

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(mask, dtype=torch.long) for mask in attention_mask],
        batch_first=True,
        padding_value=0,
    )

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if has_pairs:
        positive_response_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in positive_response_ids],
            batch_first=True,
            padding_value=0,
        )
        positive_response_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask, dtype=torch.long) for mask in positive_response_mask],
            batch_first=True,
            padding_value=0,
        )
        negative_response_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in negative_response_ids],
            batch_first=True,
            padding_value=0,
        )
        negative_response_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask, dtype=torch.long) for mask in negative_response_mask],
            batch_first=True,
            padding_value=0,
        )
        batch_dict["positive_response_ids"] = positive_response_ids
        batch_dict["positive_response_mask"] = positive_response_mask
        batch_dict["negative_response_ids"] = negative_response_ids
        batch_dict["negative_response_mask"] = negative_response_mask

    return {
        "batch": batch_dict,
        "non_tensor_batch": non_tensor_batch,
        "meta_info": {},
    }
    
def collate_fn_bak(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def collate_fn(data_list: List[Dict]) -> Dict:
    """Collate function for RLHFDataset."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        if len(val) == 0:
            raise ValueError(f"Empty data found for key: {key}")
        tensors[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True, padding_value=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {"batch": dict(tensors), "non_tensor_batch": dict(non_tensors), "meta_info": {}}


class PreferencePairDataset(Dataset):
    """Dataset class for generating preference pairs from Parquet files or Hugging Face datasets."""
    def __init__(
        self,
        data_files: Optional[Union[str, List[str]]] = None,
        dataset_name: Optional[str] = None,
        tokenizer: PreTrainedTokenizer = None,
        reward_model_name: Optional[str] = None,
        max_length: int = 512,
        cache_dir: str = "~/.cache/verl/rlhf",
        mode: str = "k_fold",
        processor: Optional[ProcessorMixin] = None,
        config: Optional[DictConfig] = None,
    ):
        """
        Initialize PreferencePairDataset.

        Args:
            data_files: Path(s) to Parquet file(s) containing the dataset.
            dataset_name: Hugging Face dataset name (e.g., "lmsys/lmsys-chat-1m").
            tokenizer: PreTrainedTokenizer for tokenizing prompts and responses.
            reward_model_name: Optional reward model for scoring responses.
            max_length: Maximum sequence length for tokenization.
            cache_dir: Directory for caching dataset.
            mode: Difficulty coefficient computation mode ("k_fold", "zpd", "feature_based").
            processor: Optional processor for multi-modal data.
            config: Configuration object with dataset settings.
        """
        if (data_files is None and dataset_name is None) or (data_files is not None and dataset_name is not None):
            raise ValueError("Exactly one of data_files or dataset_name must be provided.")

        self.data_files = [data_files] if isinstance(data_files, str) else data_files
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.reward_model = None
        if reward_model_name and torch.cuda.is_available():
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).eval().cuda()
        self.max_length = max_length
        self.mode = mode
        self.processor = processor
        self.config = config or {}
        self.cache_dir = os.path.expanduser(cache_dir)
        self.pairs = []
        self.task_features = []
        self.difficulty_coeffs = []
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.is_chatbot_arena = dataset_name == "lmsys/chatbot_arena_conversations" if dataset_name else False

        self._load_data()
        self._preprocess()

    def _load_data(self):
        """Load data from Parquet files or Hugging Face dataset."""
        if self.data_files:
            dataframes = []
            for file in self.data_files:
                file = os.path.expanduser(file)
                dataframe = datasets.load_dataset("parquet", data_files=file, cache_dir=self.cache_dir)["train"]
                dataframes.append(dataframe)
            self.dataset = datasets.concatenate_datasets(dataframes)
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]

    def compute_reward(self, text: str) -> float:
        """Compute reward score for a given text using the reward model."""
        if not self.reward_model:
            return 1.0  # Default if no reward model
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to("cuda")
        with torch.no_grad():
            reward = self.reward_model(**inputs).logits.sigmoid().item()
        return reward

    def _preprocess(self):
        """Preprocess dataset to create preference pairs."""
        if self.is_chatbot_arena:
            self._preprocess_chatbot_arena()
        elif self.data_files and "positive_response" in self.dataset.column_names:
            self._preprocess_parquet_with_pairs()
        else:
            self._preprocess_generic()

    def _preprocess_parquet_with_pairs(self):
        """Preprocess Parquet dataset with explicit positive/negative response columns."""
        for sample in self.dataset:
            prompt = sample[self.prompt_key]
            pos_response = sample["positive_response"]
            neg_response = sample["negative_response"]
            self._process_pair(prompt, pos_response, neg_response, sample.get("index", 0))

    def _preprocess_chatbot_arena(self):
        for sample in self.dataset:
            conversation_a = sample["conversation_a"]
            conversation_b = sample["conversation_b"]
            if len(conversation_a) < 2 or len(conversation_b) < 2:
                continue
            prompt = conversation_a[0]["content"]
            turn_id = sample["turn"]
            response_a = conversation_a[turn_id]["content"]
            response_b = conversation_b[turn_id]["content"]

            preference = sample['winner']
            pos_response = response_a if preference == "model_a" else response_b
            neg_response = response_b if preference == "model_a" else response_a
            self._process_pair(prompt, pos_response, neg_response, sample.get("index", 0)) 

    def _preprocess_generic(self):
        """Preprocess generic dataset by grouping responses and scoring with reward model."""
        prompt_to_responses = defaultdict(list)
        for sample in self.dataset:
            prompt = sample[self.prompt_key]
            response = sample.get("response", "")
            if not response:
                continue
            prompt_to_responses[prompt].append((response, sample.get("index", 0)))

        for prompt, responses in prompt_to_responses.items():
            if len(responses) < 2:
                continue
            response_rewards = [(resp, idx, self.compute_reward(resp)) for resp, idx in responses]
            response_rewards.sort(key=lambda x: x[2], reverse=True)
            pos_response, pos_idx, _ = response_rewards[0]
            neg_response, neg_idx, _ = response_rewards[-1]
            self._process_pair(prompt, pos_response, neg_response, pos_idx)

    def _process_pair(self, prompt: str, pos_response: str, neg_response: str, index: int):
        """Tokenize and compute difficulty coefficient for a preference pair."""
        prompt_inputs = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt", add_special_tokens=True
        )
        pos_inputs = self.tokenizer(
            pos_response, max_length=self.max_length, truncation=True, return_tensors="pt", add_special_tokens=False
        )
        neg_inputs = self.tokenizer(
            neg_response, max_length=self.max_length, truncation=True, return_tensors="pt", add_special_tokens=False
        )

        pos_mask = pos_inputs["attention_mask"].squeeze(0)
        neg_mask = neg_inputs["attention_mask"].squeeze(0)

        seq_length = (pos_inputs["input_ids"].shape[1] + neg_inputs["input_ids"].shape[1]) / 2
        reward_sparsity = 1.0  # Placeholder
        task_features = {"seq_length": seq_length, "reward_sparsity": reward_sparsity}

        difficulty_coeff = torch.tensor(1.0, device="cuda" if torch.cuda.is_available() else "cpu")
        if self.reward_model and torch.cuda.is_available():
            pos_log_probs = torch.log_softmax(self.reward_model(pos_inputs["input_ids"].cuda()).logits, dim=-1)
            pos_values = self.reward_model(pos_inputs["input_ids"].cuda()).logits
            pos_rewards = torch.tensor([self.compute_reward(pos_response)], device="cuda")
            difficulty_coeff = compute_difficulty_coeff(
                log_probs=pos_log_probs,
                values=pos_values,
                rewards=pos_rewards,
                response_mask=pos_mask.cuda(),
                task_features=task_features,
                mode=self.mode
            )

        self.pairs.append({
            "prompt": prompt_inputs["input_ids"].squeeze(0),
            "prompt_mask": prompt_inputs["attention_mask"].squeeze(0),
            "pos_response": pos_inputs["input_ids"].squeeze(0),
            "neg_response": neg_inputs["input_ids"].squeeze(0),
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
            "difficulty_coeff": difficulty_coeff,
            "prompt_text": prompt,
            "pos_response_text": pos_response,
            "neg_response_text": neg_response,
            "index": index
        })
        self.task_features.append(task_features)
        self.difficulty_coeffs.append(difficulty_coeff.item())

    def generate_curriculum(self, agent_performance: List[float], mode: str = "zpd") -> List[int]:
        """Generate curriculum based on difficulty coefficients."""
        task_pool = list(range(len(self.pairs)))
        return generate_curriculum(
            task_pool=task_pool,
            agent_performance=torch.tensor(agent_performance, dtype=torch.float),
            difficulty_coeffs=torch.tensor(self.difficulty_coeffs),
            mode=mode
        )

    def __getitem__(self, idx: int) -> Dict:
        return self.pairs[idx]

    def __len__(self) -> int:
        return len(self.pairs)


class RLHFDataset(Dataset):
    """Dataset for RLHF training, extended to support preference pairs."""
    def __init__(
        self,
        data_files: Optional[Union[str, ListConfig[str]]] = None,
        dataset_name: Optional[str] = None,
        tokenizer: PreTrainedTokenizer = None,
        config: DictConfig = None,
        processor: Optional[ProcessorMixin] = None,
        reward_model_name: Optional[str] = None,
        cache_dir: str = "~/.cache/verl/rlhf",
        max_length: int = 512,
        difficulty_mode: str = "k_fold",
    ):
        data_files = None if isinstance(data_files, ListConfig) and len(data_files) == 0 else data_files
        dataset_name = None if dataset_name == '' else dataset_name
        if (data_files is None and dataset_name is None) or (data_files is not None and dataset_name is not None):
            raise ValueError("Exactly one of data_files or dataset_name must be provided.")

        self.data_files = [data_files] if isinstance(data_files, str) else data_files
        self.original_data_files = copy.deepcopy(self.data_files) if self.data_files else None
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or DictConfig({})
        self.reward_model_name = reward_model_name
        self.cache_dir = os.path.expanduser(cache_dir)
        self.max_length = max_length
        self.difficulty_mode = difficulty_mode
        self.use_preference = self.config.get("use_preference", False)

        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.image_key = self.config.get("image_key", "images")
        self.video_key = self.config.get("video_key", "videos")
        self.max_prompt_length = self.config.get("max_prompt_length", max_length)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = self.config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.serialize_dataset = False

        if self.use_preference:
            self.preference_dataset = PreferencePairDataset(
                data_files=self.data_files,
                dataset_name=self.dataset_name,
                tokenizer=tokenizer,
                reward_model_name=reward_model_name,
                max_length=max_length,
                cache_dir=cache_dir,
                mode=difficulty_mode,
                processor=processor,
                config=config
            )
            self.data = self.preference_dataset.pairs
        else:
            self._download()
            self._read_files_and_tokenize()
            self.data = self.dataframe

        # Apply preprocessing
        # if difficulty_mode == "k_fold":
        #     print("Applying k_fold preprocessing")
        #     self.data = self._apply_k_fold(self.data)
        #     print(f"Data size after k_fold: {len(self.data)}")
        # elif difficulty_mode:
        #     print(f"Applying {difficulty_mode} preprocessing")
        #     self.data = self._apply_preprocessing(self.data, difficulty_mode)
        #     print(f"Data size after {difficulty_mode}: {len(self.data)}")

        # Validate data
        # if len(self.data) == 0:
        #     print("Warning: Dataset is empty after preprocessing")
        # else:
        #     print(f"Sample data: {self.data.head()}")


    def _apply_k_fold(self, data):
        # Placeholder: replace with actual k_fold logic
        print("Applying k_fold (placeholder)")
        return data  # Temporarily return unfiltered data

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        if data_files:
            for i, parquet_file in enumerate(data_files):
                self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        if self.dataset_name:
            self.dataframe = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]
        else:
            dataframes = []
            for parquet_file in self.data_files:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file, cache_dir=self.cache_dir)["train"]
                dataframes.append(dataframe)
            self.dataframe = datasets.concatenate_datasets(dataframes)

        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe.filter(
                lambda doc: len(self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset and self.data_files:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
            self.data = self.dataframe
        elif self.use_preference:
            self.preference_dataset = PreferencePairDataset(
                data_files=self.data_files,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,
                reward_model_name=self.reward_model_name,
                max_length=self.max_length,
                cache_dir=self.cache_dir,
                mode=self.difficulty_mode,
                processor=self.processor,
                config=self.config
            )
            self.data = self.preference_dataset.pairs
        else:
            print("Using old dataloader checkpoint, consider training from scratch for better performance.")

    def __len__(self):
        return len(self.data)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    def __getitem__(self, item):
        if self.use_preference:
            return self.data[item]
        row_dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor:
            from verl.utils.dataset.vision_utils import process_image, process_video
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            images = [process_image(image) for image in row_dict.pop(self.image_key, [])]
            videos = [process_video(video) for video in row_dict.pop(self.video_key, [])]
            if images:
                multi_modal_data["image"] = images
            if videos:
                multi_modal_data["video"] = [video.numpy() for video in videos]
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        row_dict["index"] = row_dict.get("extra_info", {}).get("index", item)

        return row_dict

    def split_pairs(self, batch: DataProto) -> Tuple[Optional[DataProto], Optional[DataProto]]:
        """Split a batch into positive and negative response pairs for DAPO."""
        if not self.use_preference or "pos_response" not in batch.batch:
            return None, None

        positive_batch_dict = {
            "input_ids": batch.batch["prompt"],
            "attention_mask": batch.batch["prompt_mask"],
            "responses": batch.batch["pos_response"],
            "response_mask": batch.batch["pos_mask"],
            "difficulty_coeff": batch.batch["difficulty_coeff"]
        }
        positive_non_tensor_batch = {
            "prompt": batch.non_tensor_batch["prompt_text"],
            "data_source": batch.non_tensor_batch.get("data_source", ["unknown"] * len(batch.batch["prompt"]))
        }
        positive_batch = DataProto(
            batch=positive_batch_dict,
            non_tensor_batch=positive_non_tensor_batch,
            meta_info=batch.meta_info.copy(),
        )

        negative_batch_dict = {
            "input_ids": batch.batch["prompt"],
            "attention_mask": batch.batch["prompt_mask"],
            "responses": batch.batch["neg_response"],
            "response_mask": batch.batch["neg_mask"],
            "difficulty_coeff": batch.batch["difficulty_coeff"]
        }
        negative_non_tensor_batch = {
            "prompt": batch.non_tensor_batch["prompt_text"],
            "data_source": batch.non_tensor_batch.get("data_source", ["unknown"] * len(batch.batch["prompt"]))
        }
        negative_batch = DataProto(
            batch=negative_batch_dict,
            non_tensor_batch=negative_non_tensor_batch,
            meta_info=batch.meta_info.copy(),
        )

        return positive_batch, negative_batch

    def generate_curriculum(self, agent_performance: List[float], mode: str = "zpd") -> List[int]:
        """Generate curriculum based on difficulty coefficients."""
        if not self.use_preference:
            return list(range(len(self)))
        task_pool = list(range(len(self.data)))
        difficulty_coeffs = [item["difficulty_coeff"].item() for item in self.data]
        return generate_curriculum(
            task_pool=task_pool,
            agent_performance=torch.tensor(agent_performance, dtype=torch.float),
            difficulty_coeffs=torch.tensor(difficulty_coeffs),
            mode=mode
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.serialize_dataset:
            if "dataframe" in state:
                del state["dataframe"]
            if "data" in state and not self.use_preference:
                del state["data"]
        return state

