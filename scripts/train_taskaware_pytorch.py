"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import contextlib # [COPILOT] for gradient accumulation
import dataclasses
import gc
import logging
import os
import platform
import random  # [COPILOT] Save/restore Python RNG state across resume.
import signal  # [COPILOT] Graceful Ctrl+C checkpoint-on-interrupt handling.
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.amp # [COPILOT]
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
from openpi.models_pytorch import pi0_pytorch, pi0_taskaware_pytorch, projectors  # [COPILOT] Use task-aware model.
from openpi.models_pytorch.lora_copilot import apply_lora, get_trainable_parameters, mark_only_lora_as_trainable # [COPILOT]
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

from vggt.models.vggt import VGGT


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig, *, batch_size: int | None = None):
    # Use the unified data loader with PyTorch framework.
    # [COPILOT] Supports stage-specific batch size by overriding TrainConfig.batch_size at runtime.
    if batch_size is not None and batch_size != config.batch_size:
        config = dataclasses.replace(config, batch_size=batch_size)
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def _get_rng_state():  # [COPILOT] Collect deterministic RNG checkpoints for resume.
    rng_state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return rng_state


def _set_rng_state(rng_state):  # [COPILOT] Restore deterministic RNG checkpoints on resume.
    # [COPILOT] Restore RNG states from checkpoint (best-effort if some keys are missing).
    if not rng_state:
        return
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if torch.cuda.is_available() and "torch_cuda_all" in rng_state:
        # [COPILOT] Handle checkpoint/runtime CUDA-device-count mismatch (e.g., different WORLD_SIZE or GPU visibility).
        cuda_states = rng_state["torch_cuda_all"]
        if isinstance(cuda_states, list | tuple) and len(cuda_states) > 0:
            current_cuda_count = len(torch.cuda.default_generators)
            if len(cuda_states) == current_cuda_count:
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                # [COPILOT] Fallback: restore only the current CUDA device state to avoid IndexError.
                current_device = torch.cuda.current_device()
                safe_idx = min(current_device, len(cuda_states) - 1)
                torch.cuda.set_rng_state(cuda_states[safe_idx], device=current_device)
                logging.warning(
                    "CUDA RNG state count mismatch (ckpt=%d, runtime=%d). Restored only device %d using state index %d.",
                    len(cuda_states),
                    current_cuda_count,
                    current_device,
                    safe_idx,
                )
        else:
            logging.warning("Invalid or empty torch_cuda_all RNG payload; skipping CUDA RNG restore.")


def save_checkpoint(model, align_projector, optimizer, scaler, global_step, config, is_main, data_config, force=False):  # [COPILOT] Save full training state including alignment/scaler/RNG.
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # [COPILOT] Save on schedule/final-step, or force-save on interrupt.
    should_save = force or (
        (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1
    )
    if should_save:
        
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # [COPILOT] Save align projector state; this was previously missing in resume.
        align_projector_to_save = (
            align_projector.module if isinstance(align_projector, torch.nn.parallel.DistributedDataParallel) else align_projector
        )
        safetensors.torch.save_model(align_projector_to_save, tmp_ckpt_dir / "align_projector.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # [COPILOT] Save AMP GradScaler state for float16 stability on resume.
        if scaler is not None:
            torch.save(scaler.state_dict(), tmp_ckpt_dir / "scaler.pt")

        # [COPILOT] Save RNG states (torch/cuda/numpy/python) for reproducibility across resume.
        torch.save(_get_rng_state(), tmp_ckpt_dir / "rng_state.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "forced": force,  # [COPILOT] Mark checkpoints created by forced interrupt save.
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, align_projector, optimizer, scaler, checkpoint_dir, device):  # [COPILOT] Load full training state including alignment/scaler/RNG.
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        # [COPILOT] Load align projector state so alignment branch resumes continuously.
        align_path = ckpt_dir / "align_projector.safetensors"
        if align_path.exists():
            align_to_load = (
                align_projector.module
                if isinstance(align_projector, torch.nn.parallel.DistributedDataParallel)
                else align_projector
            )
            safetensors.torch.load_model(align_to_load, align_path, device=str(device))
            logging.info("Loaded align projector state from safetensors format")
        else:
            logging.warning("No align_projector checkpoint found at %s; using fresh initialization", ckpt_dir)

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # [COPILOT] Load GradScaler state if available (float16 AMP only).
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler is not None and scaler_path.exists():
            scaler_state = torch.load(scaler_path, map_location=device, weights_only=False)
            scaler.load_state_dict(scaler_state)
            del scaler_state
            logging.info("Loaded GradScaler state from pt format")
        elif scaler is not None:
            logging.warning("No GradScaler checkpoint found at %s; scaler starts from fresh state", ckpt_dir)

        # [COPILOT] Restore RNG state for torch/cuda/numpy/python.
        rng_state_path = ckpt_dir / "rng_state.pt"
        if rng_state_path.exists():
            rng_state = torch.load(rng_state_path, map_location="cpu", weights_only=False)
            _set_rng_state(rng_state)
            del rng_state
            logging.info("Loaded RNG states from pt format")
        else:
            logging.warning("No RNG checkpoint found at %s; stochastic order may diverge after resume", ckpt_dir)

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # [COPILOT] Setup AMP
    use_amp = device.type == "cuda" and config.pytorch_training_precision in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if config.pytorch_training_precision == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def _to_device_with_dtype(x: torch.Tensor):
        if torch.is_tensor(x):
            if x.is_floating_point():
                x = x.to(dtype=torch.float32)
            return x.to(device)
        return x

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # [COPILOT] Runtime batch/accum can be stage-specific; world size stays fixed.
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    bootstrap_batch_cfg = getattr(config, "taskaware_stage1_batch_size", None)
    if bootstrap_batch_cfg is None:
        bootstrap_batch_size = int(os.environ.get("TASKAWARE_STAGE1_BATCH_SIZE", str(config.batch_size)))
    else:
        bootstrap_batch_size = int(bootstrap_batch_cfg)
    if bootstrap_batch_size % world_size != 0:
        raise ValueError(
            f"Initial batch size ({bootstrap_batch_size}) must be divisible by world_size ({world_size}) for DDP"
        )
    grad_accum_debug = os.environ.get("GRAD_ACCUM_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}  # [COPILOT] debug flag
    if is_main and grad_accum_debug:
        logging.info("Grad accumulation debug enabled (GRAD_ACCUM_DEBUG=1)")  # [COPILOT] DEBUG

    # [COPILOT] Initialize loader with stage1 batch fallback; runtime stage logic may rebuild later.
    loader, data_config = build_datasets(config, batch_size=bootstrap_batch_size)
    current_batch_size = int(bootstrap_batch_size)
    effective_batch_size = current_batch_size // world_size
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
    if grad_accum_steps < 1:
        raise ValueError("GRAD_ACCUM_STEPS must be >= 1")

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_config = dataclasses.replace(config, batch_size=bootstrap_batch_size)
        sample_data_loader = _data.create_data_loader(sample_config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_taskaware_pytorch.PI0Pytorch(model_cfg, config).to(device)  # [COPILOT] Task-aware alignment model.
    vggt_model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    ).to(device) # [COPILOT][DEBUG] VGGT always in float16: dtype=torch.float16
    
    # [COPILOT] VGGT is used as a frozen feature extractor.
    vggt_model.eval()
    vggt_model.requires_grad_(False)

    # [COPILOT] Use explicit task-aware projector dims from TrainConfig (no legacy fallback path).
    align_projector = projectors.AlignProjector(
        model.LLM_width,
        config.vggt_dim,
        config.use_vlm_norm,
        hidden_dim=config.align_projector_hidden_dim,
        out_dim=config.align_projector_out_dim,
    ).to(device)

    # [COPILOT] Switch gradient checkpointing from TrainConfig/CLI flag.
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            enable_gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing via config/CLI flag")
        else:
            enable_gradient_checkpointing = False
            logging.warning(
                "Requested gradient checkpointing, but model does not support it. Continuing with checkpointing disabled."
            )
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            enable_gradient_checkpointing = False
            model.gradient_checkpointing_disable()
            logging.info("Gradient checkpointing disabled (default)")
        elif hasattr(model, "gradient_checkpointing_enable"):
            enable_gradient_checkpointing = False
            logging.info("Gradient checkpointing is supported, but disabled by config")
        else:
            enable_gradient_checkpointing = False
            logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # [COPILOT] Apply LoRA if enabled (before DDP wrapping and optimizer creation)
    if config.lora_enabled:
        replaced = apply_lora(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        mark_only_lora_as_trainable(model)
        # [COPILOT] Keep all task-aware modules fully trainable even when LoRA mode freezes base model weights.
        for name, param in model.named_parameters():
            if name.startswith("taskaware_"):
                param.requires_grad = True

        # [COPILOT] Ensure vision tower does NOT participate in LoRA updates
        vision_tower_prefix = "paligemma_with_expert.paligemma.vision_tower"
        for name, param in model.named_parameters():
            if name.startswith(vision_tower_prefix) and ("lora_A" in name or "lora_B" in name):
                param.requires_grad = False

        if is_main:
            trainable = sum(p.numel() for p in get_trainable_parameters(model))
            total = sum(p.numel() for p in model.parameters())
            logging.info(
                "Enabled LoRA: replaced %d Linear layers, pre-stage trainable params=%d/%d "
                "(stage runtime gating is applied later)",
                replaced,
                trainable,
                total,
            )

    # [COPILOT] Keep FP32 weights when GradScaler is enabled; otherwise honor training precision
    if use_amp and amp_dtype == torch.float16:
        model = model.to(device=device, dtype=torch.float32)
        align_projector = align_projector.to(device=device, dtype=torch.float32)
    else:
        if config.pytorch_training_precision == "bfloat16":
            target_dtype = torch.bfloat16
        elif config.pytorch_training_precision == "float16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        model = model.to(device=device, dtype=target_dtype)
        align_projector = align_projector.to(device=device, dtype=target_dtype)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=False,  # [COPILOT] Disabled because 3-stage training changes trainable parameter groups.
        )
        align_projector = torch.nn.parallel.DistributedDataParallel(
            align_projector,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=False,  # [COPILOT] Disabled because 3-stage training changes trainable parameter groups.
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
            strict=False,
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    if config.vggt_weight_path is not None:
        vggt_path = os.path.join(config.vggt_weight_path, "model.pt")
        if not os.path.exists(vggt_path):
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_path}")
        vggt_model.load_state_dict(torch.load(vggt_path), strict=False)
        logging.info(f"Loaded VGGT weights from {config.vggt_weight_path}")

    # Optimizer + learning rate schedule from config
    base_warmup_steps = int(config.lr_schedule.warmup_steps)
    base_peak_lr = float(config.lr_schedule.peak_lr)
    base_decay_steps = int(config.lr_schedule.decay_steps)
    base_end_lr = float(config.lr_schedule.decay_lr)
    # [COPILOT][EXCLUDE] LM/ITG is disabled in the model; task-aware auxiliary objective is ITC + ITM only.
    taskaware_loss_coeff = float(getattr(config, "taskaware_loss_coeff", 1.0))
    # [COPILOT] Resolve stage settings with priority: config.py > env var > default.
    def _resolve_int_setting(config_value, env_key: str, default: int) -> int:
        if config_value is not None:
            return int(config_value)
        return int(os.environ.get(env_key, str(default)))

    def _resolve_float_setting(config_value, env_key: str, default: float) -> float:
        if config_value is not None:
            return float(config_value)
        return float(os.environ.get(env_key, str(default)))

    # [COPILOT] Fixed 3-stage schedule defaults: 5k / 13k / 2k.
    stage1_steps = _resolve_int_setting(getattr(config, "taskaware_stage1_steps", None), "TASKAWARE_STAGE1_STEPS", 5000)
    stage2_steps = _resolve_int_setting(getattr(config, "taskaware_stage2_steps", None), "TASKAWARE_STAGE2_STEPS", 13000)
    stage3_steps = _resolve_int_setting(getattr(config, "taskaware_stage3_steps", None), "TASKAWARE_STAGE3_STEPS", 2000)
    if stage1_steps < 0 or stage2_steps < 0 or stage3_steps < 0:
        raise ValueError("TASKAWARE_STAGE{1,2,3}_STEPS must be non-negative")
    if (stage2_steps > 0 or stage3_steps > 0) and not config.lora_enabled:
        raise ValueError("3-stage setup requires lora_enabled=True because stage 2/3 train LoRA adapters.")
    if stage1_steps + stage2_steps + stage3_steps != config.num_train_steps:
        raise ValueError(
            "3-stage step sum must equal num_train_steps. "
            f"Got stage sum={stage1_steps + stage2_steps + stage3_steps}, num_train_steps={config.num_train_steps}"
        )
    stage1_task_loss_coeff = _resolve_float_setting(
        getattr(config, "taskaware_stage1_task_loss_coeff", None),
        "TASKAWARE_STAGE1_TASK_LOSS_COEFF",
        1.0,
    )
    stage3_task_loss_coeff = _resolve_float_setting(
        getattr(config, "taskaware_stage3_task_loss_coeff", None),
        "TASKAWARE_STAGE3_TASK_LOSS_COEFF",
        taskaware_loss_coeff,
    )
    base_grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
    stage1_batch_size = _resolve_int_setting(
        getattr(config, "taskaware_stage1_batch_size", None),
        "TASKAWARE_STAGE1_BATCH_SIZE",
        int(config.batch_size),
    )
    stage23_batch_size = _resolve_int_setting(
        getattr(config, "taskaware_stage23_batch_size", None),
        "TASKAWARE_STAGE23_BATCH_SIZE",
        int(config.batch_size),
    )
    stage1_grad_accum_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage1_grad_accum_steps", None),
        "TASKAWARE_STAGE1_GRAD_ACCUM_STEPS",
        base_grad_accum_steps,
    )
    stage23_grad_accum_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage23_grad_accum_steps", None),
        "TASKAWARE_STAGE23_GRAD_ACCUM_STEPS",
        base_grad_accum_steps,
    )
    stage1_lr_warmup_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage1_lr_warmup_steps", None),
        "TASKAWARE_STAGE1_LR_WARMUP_STEPS",
        base_warmup_steps,
    )
    stage1_lr_peak = _resolve_float_setting(
        getattr(config, "taskaware_stage1_lr_peak", None),
        "TASKAWARE_STAGE1_LR_PEAK",
        base_peak_lr,
    )
    stage1_lr_decay_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage1_lr_decay_steps", None),
        "TASKAWARE_STAGE1_LR_DECAY_STEPS",
        base_decay_steps,
    )
    stage1_lr_end = _resolve_float_setting(
        getattr(config, "taskaware_stage1_lr_decay_lr", None),
        "TASKAWARE_STAGE1_LR_DECAY_LR",
        base_end_lr,
    )
    stage23_lr_warmup_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage23_lr_warmup_steps", None),
        "TASKAWARE_STAGE23_LR_WARMUP_STEPS",
        base_warmup_steps,
    )
    stage23_lr_peak = _resolve_float_setting(
        getattr(config, "taskaware_stage23_lr_peak", None),
        "TASKAWARE_STAGE23_LR_PEAK",
        base_peak_lr,
    )
    stage23_lr_decay_steps = _resolve_int_setting(
        getattr(config, "taskaware_stage23_lr_decay_steps", None),
        "TASKAWARE_STAGE23_LR_DECAY_STEPS",
        base_decay_steps,
    )
    stage23_lr_end = _resolve_float_setting(
        getattr(config, "taskaware_stage23_lr_decay_lr", None),
        "TASKAWARE_STAGE23_LR_DECAY_LR",
        base_end_lr,
    )

    def _validate_runtime_batch_and_accum(batch_size: int, grad_steps: int, stage_name: str) -> None:
        if batch_size <= 0:
            raise ValueError(f"{stage_name} batch_size must be > 0")
        if batch_size % world_size != 0:
            raise ValueError(f"{stage_name} batch_size ({batch_size}) must be divisible by world_size ({world_size})")
        if grad_steps < 1:
            raise ValueError(f"{stage_name} GRAD_ACCUM_STEPS must be >= 1")

    _validate_runtime_batch_and_accum(stage1_batch_size, stage1_grad_accum_steps, "stage1")
    _validate_runtime_batch_and_accum(stage23_batch_size, stage23_grad_accum_steps, "stage2/3")
    if stage1_lr_warmup_steps < 0 or stage23_lr_warmup_steps < 0:
        raise ValueError("Stage warmup steps must be non-negative")
    if stage1_lr_decay_steps <= 0 or stage23_lr_decay_steps <= 0:
        raise ValueError("Stage decay steps must be > 0")

    def _unwrap_module(m):
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    def _stage_from_step(step: int) -> int:
        if step < stage1_steps:
            return 1
        if step < stage1_steps + stage2_steps:
            return 2
        return 3

    def _stage_loss_weights(stage: int) -> tuple[float, float, float]:
        if stage == 1:
            return 0.0, 0.0, stage1_task_loss_coeff
        if stage == 2:
            return 1.0, float(config.align_loss_coeff), 0.0
        return 1.0, float(config.align_loss_coeff), stage3_task_loss_coeff

    def _stage_batch_and_accum(stage: int) -> tuple[int, int]:
        if stage == 1:
            return stage1_batch_size, stage1_grad_accum_steps
        return stage23_batch_size, stage23_grad_accum_steps

    def _cosine_lr(step: int, warmup_steps: int, peak_lr: float, decay_steps: int, end_lr: float) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    def _stage_lr(stage: int, step: int) -> float:
        if stage == 1:
            stage_step = step
            return _cosine_lr(stage_step, stage1_lr_warmup_steps, stage1_lr_peak, stage1_lr_decay_steps, stage1_lr_end)
        stage_step = max(0, step - stage1_steps)
        return _cosine_lr(
            stage_step,
            stage23_lr_warmup_steps,
            stage23_lr_peak,
            stage23_lr_decay_steps,
            stage23_lr_end,
        )

    def _configure_stage(stage: int) -> None:
        model_base = _unwrap_module(model)
        if not hasattr(model_base, "set_trainable_for_stage"):
            raise AttributeError("PI0Pytorch model must implement set_trainable_for_stage for staged training.")
        model_base.set_trainable_for_stage(stage, lora_enabled=config.lora_enabled)

        align_base = _unwrap_module(align_projector)
        align_trainable = stage in (2, 3)
        for param in align_base.parameters():
            param.requires_grad = align_trainable

    def _collect_param_stats() -> dict[str, dict[str, int]]:
        model_base = _unwrap_module(model)
        align_base = _unwrap_module(align_projector)

        stats: dict[str, dict[str, int]] = {
            "QFormer": {"total": 0, "trainable": 0},
            "VLM": {"total": 0, "trainable": 0},
            "Align": {"total": 0, "trainable": 0},
            "VGGT": {"total": 0, "trainable": 0},
        }

        for name, param in model_base.named_parameters():
            bucket = "QFormer" if name.startswith("taskaware_") else "VLM"
            n = param.numel()
            stats[bucket]["total"] += n
            if param.requires_grad:
                stats[bucket]["trainable"] += n

        for _, param in align_base.named_parameters():
            n = param.numel()
            stats["Align"]["total"] += n
            if param.requires_grad:
                stats["Align"]["trainable"] += n

        for _, param in vggt_model.named_parameters():
            n = param.numel()
            stats["VGGT"]["total"] += n
            if param.requires_grad:
                stats["VGGT"]["trainable"] += n

        stats["ALL"] = {
            "total": sum(stats[k]["total"] for k in ("QFormer", "VLM", "Align", "VGGT")),
            "trainable": sum(stats[k]["trainable"] for k in ("QFormer", "VLM", "Align", "VGGT")),
        }
        return stats

    def _validate_stage_param_stats(stage: int, stats: dict[str, dict[str, int]]) -> None:
        q_train = stats["QFormer"]["trainable"]
        vlm_train = stats["VLM"]["trainable"]
        align_train = stats["Align"]["trainable"]
        vggt_train = stats["VGGT"]["trainable"]

        if vggt_train != 0:
            raise RuntimeError(f"VGGT must stay frozen, but found {vggt_train} trainable params.")
        if stage == 1:
            if vlm_train != 0 or align_train != 0:
                raise RuntimeError(
                    f"Stage 1 requires VLM/Align frozen, but found VLM={vlm_train}, Align={align_train} trainable params."
                )
            if q_train == 0:
                raise RuntimeError("Stage 1 expects task-aware (QFormer bucket) trainable params, but found none.")
        elif stage == 2:
            if q_train != 0:
                raise RuntimeError(f"Stage 2 requires QFormer frozen, but found {q_train} trainable params.")
            if vlm_train == 0 or align_train == 0:
                raise RuntimeError(
                    f"Stage 2 expects VLM LoRA + Align trainable, but found VLM={vlm_train}, Align={align_train}."
                )
        elif stage == 3:
            if q_train == 0 or vlm_train == 0 or align_train == 0:
                raise RuntimeError(
                    "Stage 3 expects QFormer + VLM LoRA + Align trainable, "
                    f"but found QFormer={q_train}, VLM={vlm_train}, Align={align_train}."
                )

    def _log_stage_param_stats(stage: int, stats: dict[str, dict[str, int]], *, reason: str) -> None:
        if not is_main:
            return
        logging.info(
            "Param stats (%s, stage=%d): ALL trainable=%d / total=%d",
            reason,
            stage,
            stats["ALL"]["trainable"],
            stats["ALL"]["total"],
        )
        logging.info(
            "  QFormer trainable=%d / total=%d | VLM trainable=%d / total=%d | Align trainable=%d / total=%d | VGGT trainable=%d / total=%d",
            stats["QFormer"]["trainable"],
            stats["QFormer"]["total"],
            stats["VLM"]["trainable"],
            stats["VLM"]["total"],
            stats["Align"]["trainable"],
            stats["Align"]["total"],
            stats["VGGT"]["trainable"],
            stats["VGGT"]["total"],
        )

    # [COPILOT] Create optimizer with config parameters
    optim_params = list(model.parameters()) + list(align_projector.parameters())
    if config.lora_enabled:
        optim_params = get_trainable_parameters(model) + list(align_projector.parameters())

    optim = torch.optim.AdamW(
        optim_params, # [COPILOT]
        lr=base_peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        # [COPILOT] Resume all train states (model/align projector/optimizer/scaler/RNG).
        global_step = load_checkpoint(model, align_projector, optim, scaler, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    action_loss_coeff = 0.0
    align_loss_coeff = 0.0
    task_loss_coeff = 0.0
    current_stage_param_stats: dict[str, dict[str, int]] = {}

    def _apply_stage_runtime(stage: int, *, force_loader_rebuild: bool = False) -> bool:
        nonlocal loader, data_config, current_batch_size, effective_batch_size, grad_accum_steps, current_stage_param_stats

        _configure_stage(stage)
        desired_batch_size, desired_grad_accum_steps = _stage_batch_and_accum(stage)
        loader_rebuilt = force_loader_rebuild or (desired_batch_size != current_batch_size)
        if loader_rebuilt:
            loader, data_config = build_datasets(config, batch_size=desired_batch_size)
            current_batch_size = desired_batch_size
            effective_batch_size = current_batch_size // world_size
        grad_accum_steps = desired_grad_accum_steps
        current_stage_param_stats = _collect_param_stats()
        _validate_stage_param_stats(stage, current_stage_param_stats)
        return loader_rebuilt

    current_stage = _stage_from_step(global_step)
    _apply_stage_runtime(current_stage)
    action_loss_coeff, align_loss_coeff, task_loss_coeff = _stage_loss_weights(current_stage)
    if is_main:
        logging.info(
            "Stage setup: stage=%d steps=(%d,%d,%d) coeffs(action=%.2f, align=%.2f, task=%.2f) runtime(batch=%d, accum=%d)",
            current_stage,
            stage1_steps,
            stage2_steps,
            stage3_steps,
            action_loss_coeff,
            align_loss_coeff,
            task_loss_coeff,
            current_batch_size,
            grad_accum_steps,
        )
        if grad_accum_steps > 1:
            logging.info(
                "Using gradient accumulation: steps=%d (effective global batch=%d)",
                grad_accum_steps,
                current_batch_size * grad_accum_steps,
            )
        _log_stage_param_stats(current_stage, current_stage_param_stats, reason="initial_setup")

    model.train()
    align_projector.train()
    vggt_model.eval()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            "Training config: stage1(batch=%d, accum=%d), stage2/3(batch=%d, accum=%d), current(batch=%d, accum=%d), num_train_steps=%d",
            stage1_batch_size,
            stage1_grad_accum_steps,
            stage23_batch_size,
            stage23_grad_accum_steps,
            current_batch_size,
            grad_accum_steps,
            config.num_train_steps,
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            "LR schedule stage1(warmup=%d, peak=%.2e, decay_steps=%d, end=%.2e), "
            "stage2/3(warmup=%d, peak=%.2e, decay_steps=%d, end=%.2e)",
            stage1_lr_warmup_steps,
            stage1_lr_peak,
            stage1_lr_decay_steps,
            stage1_lr_end,
            stage23_lr_warmup_steps,
            stage23_lr_peak,
            stage23_lr_decay_steps,
            stage23_lr_end,
        )
        logging.info(
            "3-stage schedule: stage1=%d stage2=%d stage3=%d (total=%d)",
            stage1_steps,
            stage2_steps,
            stage3_steps,
            config.num_train_steps,
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    # [COPILOT] Gradient accumulation variables
    accum_step = 0
    running_action_loss = 0.0
    running_align_loss = 0.0
    running_task_loss = 0.0

    # [COPILOT] Graceful Ctrl+C: mark stop request, then force-save checkpoint in loop.
    stop_requested = False
    _prev_sigint_handler = signal.getsignal(signal.SIGINT)

    def _copilot_sigint_handler(signum, frame):
        # [COPILOT] Defer heavy I/O to train loop instead of doing it directly in signal handler.
        nonlocal stop_requested
        stop_requested = True
        logging.warning("SIGINT received. Stopping after current point and saving a forced checkpoint.")

    signal.signal(signal.SIGINT, _copilot_sigint_handler)  # [COPILOT] Install graceful interrupt handler.

    while global_step < config.num_train_steps:
        # [COPILOT] Honor Ctrl+C stop requests at loop boundaries.
        if stop_requested:
            break
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # [COPILOT] Honor Ctrl+C stop requests before consuming a new batch.
            if stop_requested:
                break
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # [COPILOT] Update LR once per optimizer step (start of accumulation cycle)
            if accum_step == 0:
                next_stage = _stage_from_step(global_step)
                if next_stage != current_stage:
                    current_stage = next_stage
                    loader_rebuilt = _apply_stage_runtime(current_stage)
                    action_loss_coeff, align_loss_coeff, task_loss_coeff = _stage_loss_weights(current_stage)
                    if is_main:
                        logging.info(
                            "Stage transition: step=%d -> stage=%d coeffs(action=%.2f, align=%.2f, task=%.2f) runtime(batch=%d, accum=%d)",
                            global_step,
                            current_stage,
                            action_loss_coeff,
                            align_loss_coeff,
                            task_loss_coeff,
                            current_batch_size,
                            grad_accum_steps,
                        )
                        _log_stage_param_stats(current_stage, current_stage_param_stats, reason="transition")
                    if loader_rebuilt:
                        # [COPILOT] Discard current batch from old loader and restart with rebuilt stage loader.
                        break
                current_lr = _stage_lr(current_stage, global_step)
                for pg in optim.param_groups:
                    pg["lr"] = current_lr
                if grad_accum_debug and is_main:
                    logging.info(
                        "GradAccum start: global_step=%d micro_step=1/%d lr=%.2e "
                        "effective_global_batch=%d (batch_size=%d, world_size=%d, per_gpu=%d)",
                        global_step,
                        grad_accum_steps,
                        optim.param_groups[0]["lr"],
                        current_batch_size * grad_accum_steps,
                        current_batch_size,
                        world_size,
                        effective_batch_size,
                    )  # [COPILOT] debug

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(_to_device_with_dtype, observation)  # noqa: PLW2901, [COPILOT]
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # [COPILOT] Gradient accumulation context
            is_accum_step = (accum_step + 1) % grad_accum_steps != 0
            no_sync_ctx = contextlib.nullcontext()
            if use_ddp and is_accum_step:
                no_sync_ctx = contextlib.ExitStack()
                no_sync_ctx.enter_context(model.no_sync())
                no_sync_ctx.enter_context(align_projector.no_sync())

            with no_sync_ctx:
                # Forward pass (AMP) [COPILOT]
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    action_losses, align_loss, task_loss = model(
                        observation, actions, vggt=vggt_model, align_proj=align_projector
                    )
                    # [COPILOT] Stage-specific total loss.
                    loss = (
                        action_loss_coeff * action_losses
                        + align_loss_coeff * align_loss
                        + task_loss_coeff * task_loss
                    )

                # Backward pass with gradient accumulation [COPILOT]
                scaled_loss = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            running_action_loss += action_losses.item()
            running_align_loss += align_loss.item()
            running_task_loss += task_loss.item()
            accum_step += 1
            if grad_accum_debug and is_main and accum_step < grad_accum_steps:
                logging.info(
                    "GradAccum accumulating micro_step=%d/%d (no optimizer step)",
                    accum_step,
                    grad_accum_steps,
                )  # [COPILOT] debug

            # [COPILOT] Log memory usage after backward pass (early steps only, with gradient accumulation)
            if global_step < 5 and is_main and torch.cuda.is_available() and accum_step == grad_accum_steps:
                log_memory_usage(device, global_step, "after_backward")

            # [COPILOT] Only step optimizer after grad_accum_steps
            if accum_step < grad_accum_steps:
                continue
            if grad_accum_debug and is_main:
                logging.info(
                    "GradAccum optimizer step: micro_steps=%d global_step=%d",
                    grad_accum_steps,
                    global_step,
                )  # [COPILOT] debug

            # [COPILOT] Gradient clipping
            params_to_clip = [p for p in model.parameters() if p.requires_grad]
            params_to_clip += [p for p in align_projector.parameters() if p.requires_grad]
            if scaler.is_enabled():
                scaler.unscale_(optim)
            if len(params_to_clip) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=config.optimizer.clip_gradient_norm)
            else:
                grad_norm = torch.tensor(0.0, device=device)

            # Optimizer step
            if scaler.is_enabled():
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # [COPILOT] Collect stats (per optimizer step)
            if is_main:
                avg_action_loss = running_action_loss / grad_accum_steps
                avg_align_loss = running_align_loss / grad_accum_steps
                avg_task_loss = running_task_loss / grad_accum_steps
                infos.append(
                    {
                        "action_loss": avg_action_loss,
                        "align_loss": avg_align_loss,
                        "task_loss": avg_task_loss,
                        "stage": current_stage,
                        "weighted_action_loss": action_loss_coeff * avg_action_loss,
                        "weighted_align_loss": align_loss_coeff * avg_align_loss,
                        "weighted_task_loss": task_loss_coeff * avg_task_loss,
                        "total_loss": (
                            action_loss_coeff * avg_action_loss
                            + align_loss_coeff * avg_align_loss
                            + task_loss_coeff * avg_task_loss
                        ),
                        "runtime_batch_size": current_batch_size,
                        "runtime_grad_accum_steps": grad_accum_steps,
                        "effective_global_batch_size": current_batch_size * grad_accum_steps,
                        "action_loss_coeff": action_loss_coeff,
                        "align_loss_coeff": align_loss_coeff,
                        "task_loss_coeff": task_loss_coeff,
                        "total_params_all": current_stage_param_stats["ALL"]["total"],
                        "trainable_params_all": current_stage_param_stats["ALL"]["trainable"],
                        "total_params_qformer": current_stage_param_stats["QFormer"]["total"],
                        "trainable_params_qformer": current_stage_param_stats["QFormer"]["trainable"],
                        "total_params_vlm": current_stage_param_stats["VLM"]["total"],
                        "trainable_params_vlm": current_stage_param_stats["VLM"]["trainable"],
                        "total_params_align": current_stage_param_stats["Align"]["total"],
                        "trainable_params_align": current_stage_param_stats["Align"]["trainable"],
                        "total_params_vggt": current_stage_param_stats["VGGT"]["total"],
                        "trainable_params_vggt": current_stage_param_stats["VGGT"]["trainable"],
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )
            # [COPILOT] Reset accumulation variables
            running_action_loss = 0.0
            running_align_loss = 0.0
            running_task_loss = 0.0
            accum_step = 0

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["action_loss"] for info in infos) / len(infos)
                avg_align_loss = sum(info["align_loss"] for info in infos) / len(infos)
                avg_task_loss = sum(info["task_loss"] for info in infos) / len(infos)
                avg_weighted_action_loss = sum(info["weighted_action_loss"] for info in infos) / len(infos)
                avg_weighted_align_loss = sum(info["weighted_align_loss"] for info in infos) / len(infos)
                avg_weighted_task_loss = sum(info["weighted_task_loss"] for info in infos) / len(infos)
                avg_total_loss = sum(info["total_loss"] for info in infos) / len(infos)
                latest_stage = infos[-1]["stage"]
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                latest_runtime_batch_size = infos[-1]["runtime_batch_size"]
                latest_runtime_grad_accum_steps = infos[-1]["runtime_grad_accum_steps"]
                latest_effective_global_batch = infos[-1]["effective_global_batch_size"]
                latest_action_loss_coeff = infos[-1]["action_loss_coeff"]
                latest_align_loss_coeff = infos[-1]["align_loss_coeff"]
                latest_task_loss_coeff = infos[-1]["task_loss_coeff"]
                latest_total_params_all = infos[-1]["total_params_all"]
                latest_trainable_params_all = infos[-1]["trainable_params_all"]
                latest_total_params_qformer = infos[-1]["total_params_qformer"]
                latest_trainable_params_qformer = infos[-1]["trainable_params_qformer"]
                latest_total_params_vlm = infos[-1]["total_params_vlm"]
                latest_trainable_params_vlm = infos[-1]["trainable_params_vlm"]
                latest_total_params_align = infos[-1]["total_params_align"]
                latest_trainable_params_align = infos[-1]["trainable_params_align"]
                latest_total_params_vggt = infos[-1]["total_params_vggt"]
                latest_trainable_params_vggt = infos[-1]["trainable_params_vggt"]

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} stage={latest_stage} action_loss={avg_loss:.4f} align_loss={avg_align_loss:.4f} task_loss={avg_task_loss:.4f} total_loss={avg_total_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} stage={latest_stage} action_loss={avg_loss:.4f} align_loss={avg_align_loss:.4f} task_loss={avg_task_loss:.4f} total_loss={avg_total_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                logging.info(
                    "param_split stage=%d trainable(all/qformer/vlm/align/vggt)=%d/%d/%d/%d/%d total(all/qformer/vlm/align/vggt)=%d/%d/%d/%d/%d",
                    latest_stage,
                    latest_trainable_params_all,
                    latest_trainable_params_qformer,
                    latest_trainable_params_vlm,
                    latest_trainable_params_align,
                    latest_trainable_params_vggt,
                    latest_total_params_all,
                    latest_total_params_qformer,
                    latest_total_params_vlm,
                    latest_total_params_align,
                    latest_total_params_vggt,
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "action_loss": avg_loss,
                        "align_loss": avg_align_loss,
                        "task_loss": avg_task_loss,
                        "weighted_action_loss": avg_weighted_action_loss,
                        "weighted_align_loss": avg_weighted_align_loss,
                        "weighted_task_loss": avg_weighted_task_loss,
                        "total_loss": avg_total_loss,
                        "stage": latest_stage,
                        "runtime_batch_size": latest_runtime_batch_size,
                        "runtime_grad_accum_steps": latest_runtime_grad_accum_steps,
                        "effective_global_batch_size": latest_effective_global_batch,
                        "action_loss_coeff": latest_action_loss_coeff,
                        "align_loss_coeff": latest_align_loss_coeff,
                        "task_loss_coeff": latest_task_loss_coeff,
                        "total_params_all": latest_total_params_all,
                        "trainable_params_all": latest_trainable_params_all,
                        "total_params_qformer": latest_total_params_qformer,
                        "trainable_params_qformer": latest_trainable_params_qformer,
                        "total_params_vlm": latest_total_params_vlm,
                        "trainable_params_vlm": latest_trainable_params_vlm,
                        "total_params_align": latest_total_params_align,
                        "trainable_params_align": latest_trainable_params_align,
                        "total_params_vggt": latest_total_params_vggt,
                        "trainable_params_vggt": latest_trainable_params_vggt,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, align_projector, optim, scaler, global_step, config, is_main, data_config)  # [COPILOT] Periodic full-state checkpoint.

            # [COPILOT] Update progress bar with grad accumulation
            if pbar is not None:
                pbar.update(1)
                # [COPILOT] Show weighted loss breakdown terms directly in the progress bar.
                weighted_action_loss = action_loss_coeff * avg_action_loss
                weighted_align_loss = align_loss_coeff * avg_align_loss
                weighted_task_loss = task_loss_coeff * avg_task_loss
                total_loss = weighted_action_loss + weighted_align_loss + weighted_task_loss
                pbar.set_postfix(
                    {
                        "stage": current_stage,
                        "loss": f"{total_loss:.4f}",
                        "action_loss": f"{weighted_action_loss:.4f}",
                        "align_loss": f"{weighted_align_loss:.4f}",
                        "task_loss": f"{weighted_task_loss:.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )

    # [COPILOT] If stopped by Ctrl+C, force-save current state immediately.
    if stop_requested:
        save_checkpoint(model, align_projector, optim, scaler, global_step, config, is_main, data_config, force=True)  # [COPILOT] Forced full-state checkpoint on Ctrl+C.

    # [COPILOT] Restore previous SIGINT handler before teardown.
    signal.signal(signal.SIGINT, _prev_sigint_handler)  # [COPILOT] Restore previous interrupt handler.

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
