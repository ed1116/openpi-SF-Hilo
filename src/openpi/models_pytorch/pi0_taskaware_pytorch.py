import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from vggt.utils.load_fn import preprocess_images_from_openpi


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class TaskAwareQFormerLayer(nn.Module):
    # [COPILOT] One Q-Former block: bidirectional query-text self-attention + query-to-VGGT cross-attention + FFN.
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        layer_idx: int,
        cross_attention_freq: int,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # [COPILOT] BLIP-2 style scheduling: enable cross-attention on layers that satisfy layer_idx % freq == 0.
        self.has_cross_attention = layer_idx % cross_attention_freq == 0
        if self.has_cross_attention:
            # [COPILOT] Keep Q-Former hidden size at `hidden_dim` while accepting VGGT key/value tokens at `encoder_dim`.
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=encoder_dim,
                vdim=encoder_dim,
            )
        else:
            self.cross_attn = None
        self.norm_qt = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        # [COPILOT] Match BLIP-2 Q-Former style: separate FFN paths for query and text streams.
        self.norm_ffn_query = nn.LayerNorm(hidden_dim)
        self.ffn_query = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.norm_ffn_text = nn.LayerNorm(hidden_dim)
        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        vggt_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [COPILOT] Run bidirectional self-attention over [queries; text], then query-only cross-attention to VGGT.
        target_dtype = query_tokens.dtype
        if text_tokens is None or text_tokens.shape[1] == 0:
            query_len = query_tokens.shape[1]
            qt_tokens = query_tokens
            key_padding_mask = None
            has_text = False
        else:
            query_len = query_tokens.shape[1]
            # [COPILOT] Keep dtypes aligned for stable attention math under mixed-precision setups.
            text_tokens = text_tokens.to(dtype=target_dtype)
            qt_tokens = torch.cat([query_tokens, text_tokens], dim=1)
            if text_mask is None:
                text_valid = torch.ones(
                    text_tokens.shape[0], text_tokens.shape[1], dtype=torch.bool, device=text_tokens.device
                )
            else:
                text_valid = text_mask.bool()
            query_valid = torch.ones(
                text_tokens.shape[0], query_len, dtype=torch.bool, device=text_tokens.device
            )
            key_padding_mask = torch.cat([~query_valid, ~text_valid], dim=1)
            has_text = True

        qt_norm = self.norm_qt(qt_tokens)
        qt_attn, _ = self.self_attn(
            qt_norm,
            qt_norm,
            qt_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        qt_tokens = qt_tokens + self.dropout(qt_attn)

        query_tokens = qt_tokens[:, :query_len, :]
        text_tokens = qt_tokens[:, query_len:, :] if has_text else None

        if self.has_cross_attention:
            query_norm = self.norm_cross(query_tokens)
            # [COPILOT] Keep dtypes aligned for query-to-VGGT cross-attention.
            vggt_tokens = vggt_tokens.to(dtype=target_dtype)
            query_cross, _ = self.cross_attn(query_norm, vggt_tokens, vggt_tokens, need_weights=False)
            query_tokens = query_tokens + self.dropout(query_cross)

        # [COPILOT] Query stream: self-attn -> cross-attn -> query-FFN.
        query_tokens = query_tokens + self.dropout(self.ffn_query(self.norm_ffn_query(query_tokens)))
        # [COPILOT] Text stream: self-attn -> text-FFN.
        if has_text and text_tokens is not None and text_tokens.shape[1] > 0:
            text_tokens = text_tokens + self.dropout(self.ffn_text(self.norm_ffn_text(text_tokens)))
        return query_tokens, text_tokens


class TaskAwareQFormer(nn.Module):
    # [COPILOT] Lightweight Q-Former stack for task-aware VGGT token compression into fixed 16x16 queries.
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attention_freq: int = 2,
    ) -> None:
        super().__init__()
        # [COPILOT] Keep cross-attention frequency valid and configurable from TrainConfig.
        cross_attention_freq = max(1, int(cross_attention_freq))
        self.layers = nn.ModuleList(
            [
                TaskAwareQFormerLayer(
                    hidden_dim=hidden_dim,
                    encoder_dim=encoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    layer_idx=layer_idx,
                    cross_attention_freq=cross_attention_freq,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        vggt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # [COPILOT] Iteratively update queries conditioned on text and VGGT, then return normalized query tokens.
        for layer in self.layers:
            query_tokens, text_tokens = layer(query_tokens, text_tokens, text_mask, vggt_tokens)
        return self.norm(query_tokens)


class PI0Pytorch(nn.Module):
    def __init__(self, config, extra_config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.LLM_width = paligemma_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Specific config for SpatialForcing alignment
        self.vla_layers_align = extra_config.vla_layers_align
        self.vggt_layers_align = extra_config.vggt_layers_align

        # [COPILOT] Task-aware alignment config: text-conditioned Q-Former over VGGT tokens.
        self.taskaware_text_layer = getattr(extra_config, "taskaware_text_layer", None)
        if self.taskaware_text_layer is None:
            self.taskaware_text_layer = max(0, self.vla_layers_align - 1)
        self.taskaware_text_detach = bool(getattr(extra_config, "taskaware_text_detach", True))
        self.taskaware_text_dropout = float(getattr(extra_config, "taskaware_text_dropout", 0.1))
        self.taskaware_query_side = int(getattr(extra_config, "taskaware_query_side", 16))
        self.taskaware_queries_per_view = self.taskaware_query_side * self.taskaware_query_side
        self.taskaware_num_views = len(_preprocessing.IMAGE_KEYS)
        # [COPILOT] Configurable BLIP-2 style cross-attention cadence for task-aware Q-Former layers.
        self.taskaware_qformer_cross_attention_freq = max(
            1, int(getattr(extra_config, "taskaware_qformer_cross_attention_freq", 2))
        )
        # [COPILOT] Keep Q-Former query/text space at 768 and only project VLM text into that space.
        self.taskaware_qformer_dim = int(getattr(extra_config, "taskaware_qformer_dim", 768))
        self.taskaware_qformer_heads = int(getattr(extra_config, "taskaware_qformer_heads", 8))
        self.taskaware_qformer_dropout = float(getattr(extra_config, "taskaware_qformer_dropout", 0.0))
        self.taskaware_qformer = TaskAwareQFormer(
            hidden_dim=self.taskaware_qformer_dim,
            encoder_dim=self.LLM_width,
            num_heads=self.taskaware_qformer_heads,
            num_layers=int(getattr(extra_config, "taskaware_qformer_layers", 4)),
            mlp_ratio=float(getattr(extra_config, "taskaware_qformer_mlp_ratio", 4.0)),
            dropout=self.taskaware_qformer_dropout,
            cross_attention_freq=self.taskaware_qformer_cross_attention_freq,
        )
        # [COPILOT] Single-layer MLP projection: VLM text hidden (2048) -> Q-Former text input (768).
        self.taskaware_text_in_proj = nn.Linear(self.LLM_width, self.taskaware_qformer_dim)
        self.taskaware_query_tokens = nn.Parameter(
            torch.zeros(self.taskaware_num_views, self.taskaware_queries_per_view, self.taskaware_qformer_dim)
        )
        # [COPILOT] Use learnable 16x16 patch position embedding for per-view queries (no separate view embedding).
        self.taskaware_query_pos = nn.Parameter(torch.zeros(1, self.taskaware_queries_per_view, self.taskaware_qformer_dim))
        nn.init.normal_(self.taskaware_query_tokens, std=0.02)
        nn.init.normal_(self.taskaware_query_pos, std=0.02)
        # [COPILOT] BLIP-2 style auxiliary heads for task-aware ITC/ITM/LM objectives.
        self.taskaware_vision_proj = nn.Linear(self.taskaware_qformer_dim, self.taskaware_qformer_dim)
        self.taskaware_text_proj = nn.Linear(self.taskaware_qformer_dim, self.taskaware_qformer_dim)
        self.taskaware_itm_cross_attn = nn.MultiheadAttention(
            self.taskaware_qformer_dim,
            self.taskaware_qformer_heads,
            dropout=self.taskaware_qformer_dropout,
            batch_first=True,
        )
        self.taskaware_itm_norm = nn.LayerNorm(self.taskaware_qformer_dim)
        self.taskaware_itm_head = nn.Linear(self.taskaware_qformer_dim, 2)
        self.taskaware_temp = nn.Parameter(torch.tensor(0.07))

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def set_trainable_for_stage(self, stage: int, *, lora_enabled: bool) -> None:
        """Configure trainable parameter groups for 3-stage training."""
        if stage not in (1, 2, 3):
            raise ValueError(f"Unsupported stage={stage}. Expected one of (1, 2, 3).")

        # Start from full freeze, then selectively unfreeze trainable groups per stage.
        for _, param in self.named_parameters():
            param.requires_grad = False

        # Stage 1 + 3: task-aware blocks full fine-tuning.
        if stage in (1, 3):
            for name, param in self.named_parameters():
                if name.startswith("taskaware_"):
                    param.requires_grad = True

        # Stage 2 + 3: LoRA updates on VLM backbone/action expert, excluding vision tower.
        if stage in (2, 3):
            if not lora_enabled:
                raise ValueError(
                    "Stage 2/3 requires LoRA-enabled training for VLM backbone/action expert adapters."
                )
            vision_tower_prefix = "paligemma_with_expert.paligemma.vision_tower"
            for name, param in self.named_parameters():
                if ("lora_A" in name or "lora_B" in name) and not name.startswith(vision_tower_prefix):
                    param.requires_grad = True

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True, get_wo_aug=False):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train, get_wo_aug=get_wo_aug)
        return (
            list(observation.images.values()),
            list(observation.img_wo_aug.values()) if get_wo_aug else None,
            list(observation.image_padding_mask.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        img_len = len(att_masks)

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, img_len

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _prepare_taskaware_text(
        self, all_hidden_states, img_len: int, lang_masks: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # [COPILOT] Build text conditioning tokens/masks from the k-th VLM hidden state for task-aware Q-Former.
        text_layer = self.taskaware_text_layer
        if text_layer is None:
            return None, None

        (prefix_hidden_k, _) = all_hidden_states[text_layer]
        text_hidden = prefix_hidden_k[:, img_len:, :]
        text_hidden = text_hidden[:, : lang_masks.shape[1], :]
        text_mask = lang_masks.bool()

        if self.taskaware_text_detach:
            text_hidden = text_hidden.detach()

        # [COPILOT] Project VLM text tokens from 2048 -> 768 before feeding TaskAware Q-Former.
        text_hidden = self.taskaware_text_in_proj(text_hidden)

        if self.training and self.taskaware_text_dropout > 0.0:
            keep_text = (
                torch.rand(text_hidden.shape[0], 1, device=text_hidden.device) > self.taskaware_text_dropout
            )
            text_mask = text_mask & keep_text

        return text_hidden, text_mask

    def _taskaware_query_tokens(
        self,
        vggt_hidden: torch.Tensor,
        text_hidden: torch.Tensor | None,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # [COPILOT] Convert per-view VGGT tokens into fixed 16x16 query tokens using text-aware Q-Former blocks.
        bsz, num_views, _, vggt_dim = vggt_hidden.shape
        if num_views != self.taskaware_num_views:
            raise ValueError(
                f"Expected {self.taskaware_num_views} camera views, got {num_views}. "
                "Update taskaware_num_views if camera setup changed."
            )
        if vggt_dim != self.LLM_width:
            raise ValueError(f"Expected hidden dim {self.LLM_width}, got {vggt_dim}.")

        base_queries = self.taskaware_query_tokens.unsqueeze(0).expand(bsz, -1, -1, -1)
        # [COPILOT] Keep strict patch-order correspondence: query token + patch position embedding only.
        queries = base_queries + self.taskaware_query_pos[:, None, :, :]

        query_outputs = []
        for view_idx in range(num_views):
            query_outputs.append(
                self.taskaware_qformer(
                    query_tokens=queries[:, view_idx, :, :],
                    text_tokens=text_hidden,
                    text_mask=text_mask,
                    vggt_tokens=vggt_hidden[:, view_idx, :, :],
                )
            )
        query_outputs = torch.stack(query_outputs, dim=1)
        return query_outputs.reshape(bsz, num_views * self.taskaware_queries_per_view, self.taskaware_qformer_dim)

    def _taskaware_pool_text(self, text_hidden: torch.Tensor | None, text_mask: torch.Tensor | None) -> torch.Tensor:
        # [COPILOT] Mask-aware text pooling used by ITC objective.
        if text_hidden is None or text_hidden.shape[1] == 0:
            return self.taskaware_query_tokens.new_zeros((1, self.taskaware_qformer_dim))
        if text_mask is None:
            valid_mask = torch.ones(
                text_hidden.shape[0], text_hidden.shape[1], dtype=torch.bool, device=text_hidden.device
            )
        else:
            valid_mask = text_mask.bool()
        valid_mask_f = valid_mask.to(dtype=text_hidden.dtype).unsqueeze(-1)
        denom = valid_mask_f.sum(dim=1).clamp_min(1.0)
        return (text_hidden * valid_mask_f).sum(dim=1) / denom

    def _taskaware_itc_itm_losses(
        self,
        taskaware_hidden: torch.Tensor,
        text_hidden: torch.Tensor | None,
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [COPILOT] BLIP-2-inspired ITC/ITM auxiliary losses adapted to task-aware query tokens.
        bsz = taskaware_hidden.shape[0]
        if bsz == 0:
            zero = taskaware_hidden.new_zeros(())
            return zero, zero

        pooled_text = self._taskaware_pool_text(text_hidden, text_mask)
        if pooled_text.shape[0] != bsz:
            pooled_text = pooled_text.expand(bsz, -1)

        image_feats = F.normalize(self.taskaware_vision_proj(taskaware_hidden), dim=-1)
        text_feat = F.normalize(self.taskaware_text_proj(pooled_text), dim=-1)

        temp = self.taskaware_temp.clamp(min=1e-3, max=1.0)
        sim_q2t = torch.matmul(image_feats, text_feat.transpose(0, 1))
        sim_i2t = sim_q2t.max(dim=1).values / temp
        sim_t2q = torch.einsum("bd,cqd->bcq", text_feat, image_feats)
        sim_t2i = sim_t2q.max(dim=-1).values / temp

        targets = torch.arange(bsz, device=taskaware_hidden.device, dtype=torch.long)
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        if bsz < 2:
            return loss_itc, taskaware_hidden.new_zeros(())

        with torch.no_grad():
            weights_i2t = sim_i2t.detach().clone()
            weights_t2i = sim_t2i.detach().clone()
            weights_i2t.fill_diagonal_(-10000.0)
            weights_t2i.fill_diagonal_(-10000.0)
            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)
            neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze(1)
            neg_img_idx = torch.multinomial(weights_t2i, 1).squeeze(1)

        def _itm_logits(
            image_tokens: torch.Tensor,
            text_tokens: torch.Tensor | None,
            text_tokens_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            if text_tokens is None or text_tokens.shape[1] == 0:
                fused = image_tokens
            else:
                text_tokens = text_tokens.to(dtype=image_tokens.dtype)
                if text_tokens_mask is None:
                    valid_mask = torch.ones(
                        text_tokens.shape[0],
                        text_tokens.shape[1],
                        dtype=torch.bool,
                        device=text_tokens.device,
                    )
                else:
                    valid_mask = text_tokens_mask.bool()
                key_padding_mask = ~valid_mask
                itm_cross, _ = self.taskaware_itm_cross_attn(
                    image_tokens,
                    text_tokens,
                    text_tokens,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                fused = self.taskaware_itm_norm(image_tokens + itm_cross)
            return self.taskaware_itm_head(fused).mean(dim=1)

        if text_hidden is not None:
            neg_text_hidden = text_hidden[neg_text_idx]
            neg_text_mask = text_mask[neg_text_idx] if text_mask is not None else None
        else:
            neg_text_hidden = None
            neg_text_mask = None

        logits_pos = _itm_logits(taskaware_hidden, text_hidden, text_mask)
        logits_neg_img = _itm_logits(taskaware_hidden[neg_img_idx], text_hidden, text_mask)
        logits_neg_text = _itm_logits(taskaware_hidden, neg_text_hidden, neg_text_mask)
        itm_logits = torch.cat([logits_pos, logits_neg_img, logits_neg_text], dim=0)
        itm_labels = torch.cat(
            [
                torch.ones(bsz, dtype=torch.long, device=taskaware_hidden.device),
                torch.zeros(2 * bsz, dtype=torch.long, device=taskaware_hidden.device),
            ],
            dim=0,
        )
        loss_itm = F.cross_entropy(itm_logits, itm_labels)
        return loss_itc, loss_itm

    def _taskaware_lm_loss(
        self,
        all_hidden_states,
        img_len: int,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> torch.Tensor:
        # [COPILOT][EXCLUDE] LM/ITG path disabled.
        # if lang_tokens.shape[1] < 2:
        #     return lang_tokens.new_zeros((), dtype=torch.float32)
        #
        # (prefix_hidden_final, _) = all_hidden_states[-1]
        # text_hidden_final = prefix_hidden_final[:, img_len:, :]
        # text_hidden_final = text_hidden_final[:, : lang_tokens.shape[1], :]
        #
        # lm_head = self.paligemma_with_expert.paligemma.lm_head
        # logits = lm_head(text_hidden_final.to(dtype=lm_head.weight.dtype)).to(dtype=torch.float32)
        # shift_logits = logits[:, :-1, :]
        # shift_labels = lang_tokens[:, 1:].to(dtype=torch.long)
        # shift_mask = lang_masks[:, 1:].bool()
        #
        # if not torch.any(shift_mask):
        #     return logits.new_zeros(())
        #
        # token_loss = F.cross_entropy(
        #     shift_logits.reshape(-1, shift_logits.shape[-1]),
        #     shift_labels.reshape(-1),
        #     reduction="none",
        #     label_smoothing=0.1,
        # ).view_as(shift_labels)
        # shift_mask_f = shift_mask.to(dtype=token_loss.dtype)
        # return (token_loss * shift_mask_f).sum() / shift_mask_f.sum().clamp_min(1.0)
        return lang_tokens.new_zeros((), dtype=torch.float32)

    def forward(self, observation, actions, vggt, align_proj, noise=None, time=None) -> tuple[Tensor, Tensor, Tensor]:
        """Do a full training forward pass and return (action_loss, align_loss, taskaware_aux_loss)."""
        images, img_wo_aug, img_padding_mask, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True, get_wo_aug=True
        )
        img_resize_wo_aug = preprocess_images_from_openpi(img_wo_aug)  # specific for VGGT with 518px input

        # =================================== VLA action loss ===================================

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _, all_hidden_states = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=True,
            )
            return suffix_out, all_hidden_states

        suffix_out, all_hidden_states = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        action_loss = F.mse_loss(u_t, v_t)

        # =================================== Alignment loss ===================================

        # [COPILOT] Use (k+1)-th VLM image tokens as alignment target.
        (prefix_hidden, _) = all_hidden_states[self.vla_layers_align]  # 18 total layers of paligemma
        vision_hidden = prefix_hidden[:, :img_len, :]
        # [COPILOT] Use k-th VLM text tokens for task-aware conditioning in Q-Former.
        text_hidden, text_mask = self._prepare_taskaware_text(all_hidden_states, img_len, lang_masks)

        # [COPILOT] VGGT hidden states (37x37 tokens per view) used as key/value for Q-Former cross-attention.
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            vggt_output = vggt(img_resize_wo_aug)
        agg_vggt_hidden = vggt_output["features"][self.vggt_layers_align]  # 24 for total layers of VGGT
        patch_start_idx = vggt_output["patch_start_idx"]
        vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]

        # [COPILOT] Replace bilinear pooling with task-aware Q-Former compression into 16x16=256 queries per view.
        taskaware_hidden = self._taskaware_query_tokens(vggt_hidden, text_hidden, text_mask)

        # [COPILOT] Compare raw Q-Former outputs directly against projected VLM vision features.
        if taskaware_hidden.shape[:2] != vision_hidden.shape[:2]:
            raise ValueError(
                f"Task-aware hidden shape {taskaware_hidden.shape} does not match vision hidden {vision_hidden.shape}."
            )

        # [COPILOT] Empty-image feature masks for alignment loss.
        tokens_per_img = taskaware_hidden.shape[1] // len(images)
        img_masks_stack = torch.stack(img_masks, dim=1)
        align_mask = torch.repeat_interleave(img_masks_stack, repeats=tokens_per_img, dim=1)

        # [COPILOT] Remove non-rectangular padded regions from the alignment loss.
        img_padding_mask = torch.stack(img_padding_mask, dim=1)
        target_size = img_padding_mask.shape[-1] // 14  # 224/14, where 14 is the patch size of Gemma encoder
        mask_downsampled = F.interpolate(
            img_padding_mask.float(), 
            size=(target_size, target_size), 
            mode='nearest'
        ).bool().flatten(start_dim=1)
        assert align_mask.shape == mask_downsampled.shape, \
            "align_mask shape don't match img_padding_mask shape, please manually modify the patch size of Gemma encoder (now is 14)"
        align_mask = mask_downsampled & align_mask

        # [COPILOT] Project VLM hidden states and compute cosine alignment loss against task-aware VGGT queries.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            align_loss = align_proj(vision_hidden, taskaware_hidden, align_mask)

        # [COPILOT][EXCLUDE] LM/ITG objective disabled; use only ITC + ITM.
        loss_itc, loss_itm = self._taskaware_itc_itm_losses(taskaware_hidden, text_hidden, text_mask)
        # loss_lm = self._taskaware_lm_loss(all_hidden_states, img_len, lang_tokens, lang_masks)
        # taskaware_aux_loss = loss_itc + loss_itm + loss_lm
        taskaware_aux_loss = loss_itc + loss_itm

        return action_loss, align_loss, taskaware_aux_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, _, _, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
