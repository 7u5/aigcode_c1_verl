    def get_model_and_optim2(self, config):
        model = self.get_model_and_optimizer_state()
        if isinstance(model, list):
            model = model[0]
        optim_config = init_megatron_optim_config(config.actor.optim)
        optimizer = get_megatron_optimizer(model=model, config=optim_config)
        return optimizer, model
       
    
    def get_model_and_optim3(self, config):
        from megatron.core import mpu
        env_vars = {k: v for k, v in os.environ.items() if k in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']}
        print("get_model_and_optim: Environment variables:", env_vars)
        print("get_model_and_optim: torch.distributed initialized:", torch.distributed.is_initialized())
        print("get_model_and_optim: Available GPUs:", torch.cuda.device_count())
        print("get_model_and_optim: actor_rollout_wg attributes:", dir(self.actor_rollout_wg))

        # Validate distributed context
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed is not initialized. Ensure TaskRunner or NVMegatronRayWorkerGroup sets up the distributed context.")

        # Validate world_size
        world_size = torch.distributed.get_world_size()
        required_gpus = config.actor.megatron.tensor_model_parallel_size * config.actor.megatron.pipeline_model_parallel_size
        if world_size != required_gpus:
            raise ValueError(f"world_size ({world_size}) does not match required GPUs ({required_gpus}) for tensor_model_parallel_size={config.actor.megatron.tensor_model_parallel_size} and pipeline_model_parallel_size={config.actor.megatron.pipeline_model_parallel_size}")

        # Initialize Megatron parallelism
        print("get_model_and_optim: Initializing Megatron model parallel")
        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=config.actor.megatron.pipeline_model_parallel_size
            )

        # Create model provider and initialize model
        model_provider = self.create_model_provider(self.config.actor_rollout_ref)
        model = get_model(
            model_provider_func=model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            use_distributed_optimizer=True
        )

        # Handle pipeline parallelism
        if isinstance(model, list):
            model = model[0]

        # Initialize optimizer
        optim_config = init_megatron_optim_config(self.config.actor_rollout_ref.actor.optim)
        optimizer = get_megatron_optimizer(model=model, config=optim_config)
        return optimizer, model
    
    def get_model_and_optim_bad(self, config):
        # Initialize optimizer config
        optim_config = init_megatron_optim_config(config.actor.optim)
        def megatron_value_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_gptmodel_from_config
            parallel_model = get_parallel_gptmodel_from_config(
                hf_to_mcore_config(config, torch.bfloat16), config, pre_process, post_process, share_embeddings_and_output_weights=False, value=True
            )
            parallel_model.cuda()
            return parallel_model
        
        model = self.model if self.model is not None else get_model(
            model_provider_func=megatron_value_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
        )
        
        # Handle pipeline parallelism: select the first model if a list is returned
        if isinstance(model, list):
            model = model[0]
            
        optimizer = get_megatron_optimizer(model=model, config=optim_config)
        return optimizer, model
     
     
     
    def get_model_and_optim_backup(self, config):
        # Assume config is defined (Megatron configuration)
        # Assume self.actor_rollout_wg is a list of Ray actors
        global_state_dict = aggregate_state_dicts(self.actor_rollout_wg, self.get_model_config())

        # Create a non-sharded model for meta-learning
        meta_model = GPTModel(config)
        meta_model.load_state_dict(global_state_dict)
        if isinstance(meta_model, list):
            meta_model = meta_model[0]
        # Create meta-optimizer
        optim_config = init_megatron_optim_config(self.config.actor_rollout_ref.actor.optim)
        optimizer = get_megatron_optimizer(model=meta_model, config=optim_config)
        return optimizer, meta_model

        # Meta-learning loop (simplified)
        '''
        for meta_task_data, tasks in meta_learning_dataset:
            # Inner-loop training
            task_state_dicts = []
            for actor, task_data in zip(self.actor_rollout_wg, tasks):
                @ray.remote
                def inner_loop(actor, task_data):
                    model = ray.get(actor.get_model.remote())
                    optimizer = optim.Adam(model.parameters(), lr=1e-4)
                    for data in task_data:
                        loss = compute_loss(model, data)  # Define compute_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    return ray.get(actor.get_state_dict.remote())
                task_state_dicts.append(ray.get(inner_loop.remote(actor, task_data)))

            # Aggregate updated parameters
            global_state_dict = aggregate_state_dicts(task_state_dicts, config)
            meta_model.load_state_dict(global_state_dict)
        '''
    
    def get_optim(self, config):
        from verl.utils.model import get_parallel_gptmodel_from_config
        meta_model = get_parallel_gptmodel_from_config()
        optim_config = init_megatron_optim_config(self.config.actor_rollout_ref.actor.optim)
        optimizer = get_megatron_optimizer(model=meta_model, config=optim_config)
        return optimizer, meta_model        
