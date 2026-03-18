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


class NewQFormerLayer(nn.Module):
    # [COPILOT] Custom Q-Former layer using nn.MultiheadAttention (no explicit attention_scores tensor path).
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_heads: int,
        dropout: float,
        *,
        layer_idx: int,
        cross_attention_freq: int,
    ) -> None:
        super().__init__()
        mlp_hidden = hidden_dim * 4

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_ln = nn.LayerNorm(hidden_dim)

        self.has_cross_attention = layer_idx % cross_attention_freq == 0
        if self.has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=encoder_dim,
                vdim=encoder_dim,
            )
            self.cross_attn_ln = nn.LayerNorm(hidden_dim)
        else:
            self.cross_attn = None
            self.cross_attn_ln = None

        # [COPILOT] Query/text FFN are split to mirror BLIP-2 Q-Former query/text branches.
        self.query_fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.query_fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.query_ffn_ln = nn.LayerNorm(hidden_dim)

        self.text_fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.text_fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.text_ffn_ln = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_len = query_tokens.shape[1]
        if text_tokens is not None and text_tokens.shape[1] > 0:
            if text_mask is None:
                text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
            hidden_states = torch.cat([query_tokens, text_tokens.to(dtype=query_tokens.dtype)], dim=1)
            query_valid = torch.ones(query_tokens.shape[:2], dtype=torch.bool, device=query_tokens.device)
            self_key_padding_mask = torch.cat([~query_valid, ~text_mask.bool()], dim=1)
        else:
            hidden_states = query_tokens
            self_key_padding_mask = None

        self_attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.self_attn_ln(hidden_states + self.dropout(self_attn_out))

        query_tokens = hidden_states[:, :query_len, :]
        text_tokens = hidden_states[:, query_len:, :] if hidden_states.shape[1] > query_len else None

        if self.has_cross_attention:
            cross_key_padding = None
            if encoder_attention_mask is not None:
                cross_key_padding = ~encoder_attention_mask.bool()
            cross_out, _ = self.cross_attn(
                query_tokens,
                encoder_hidden_states.to(dtype=query_tokens.dtype),
                encoder_hidden_states.to(dtype=query_tokens.dtype),
                key_padding_mask=cross_key_padding,
                need_weights=False,
            )
            query_tokens = self.cross_attn_ln(query_tokens + self.dropout(cross_out))

        query_ffn = self.query_fc2(F.gelu(self.query_fc1(query_tokens)))
        query_tokens = self.query_ffn_ln(query_tokens + self.dropout(query_ffn))

        if text_tokens is not None and text_tokens.shape[1] > 0:
            text_ffn = self.text_fc2(F.gelu(self.text_fc1(text_tokens)))
            text_tokens = self.text_ffn_ln(text_tokens + self.dropout(text_ffn))

        return query_tokens, text_tokens


class NewQFormer(nn.Module):
    # [COPILOT] Custom full-size Q-Former core (12-layer capable) with MHA blocks.
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        cross_attention_freq: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cross_attention_freq = max(1, int(cross_attention_freq))

        # [COPILOT] Mirrors BLIP-2 embedding LayerNorm+Dropout behavior for dense [query;text] inputs.
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                NewQFormerLayer(
                    hidden_dim=hidden_dim,
                    encoder_dim=encoder_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_idx=layer_idx,
                    cross_attention_freq=self.cross_attention_freq,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_len = query_tokens.shape[1]
        if text_tokens is not None and text_tokens.shape[1] > 0:
            hidden_states = torch.cat([query_tokens, text_tokens.to(dtype=query_tokens.dtype)], dim=1)
            hidden_states = self.input_dropout(self.input_ln(hidden_states))
            query_tokens = hidden_states[:, :query_len, :]
            text_tokens = hidden_states[:, query_len:, :]
        else:
            query_tokens = self.input_dropout(self.input_ln(query_tokens))
            text_tokens = None

        for layer in self.layers:
            query_tokens, text_tokens = layer(
                query_tokens=query_tokens,
                text_tokens=text_tokens,
                text_mask=text_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        return query_tokens


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

        # [COPILOT] New BLIP-2-style defaults:
        # SUBTEXT from VLA layer 11, SUBIMAGE from VLA layer 12.
        new_subtext_layer = getattr(extra_config, "new_subtext_layer", None)
        new_subimage_layer = getattr(extra_config, "new_subimage_layer", None)
        self.new_subtext_layer = 11 if new_subtext_layer is None else int(new_subtext_layer)
        self.new_subimage_layer = 12 if new_subimage_layer is None else int(new_subimage_layer)
        self.new_query_side = int(getattr(extra_config, "new_query_side", 16))
        self.new_queries_per_view = self.new_query_side * self.new_query_side
        self.new_num_views = len(_preprocessing.IMAGE_KEYS)
        if self.new_num_views != 2:
            raise ValueError(
                f"new pipeline expects exactly 2 views, got {self.new_num_views} (IMAGE_KEYS={_preprocessing.IMAGE_KEYS})"
            )
        self.new_total_queries = self.new_num_views * self.new_queries_per_view

        # [COPILOT] Match pretrained BLIP-2 Q-Former core hyperparameters.
        self.new_qformer_dim = int(getattr(extra_config, "new_qformer_dim", 768))
        self.new_qformer_encoder_width = int(getattr(extra_config, "new_qformer_encoder_width", 1408))
        self.new_qformer_layers = int(getattr(extra_config, "new_qformer_layers", 12))
        self.new_qformer_heads = int(getattr(extra_config, "new_qformer_heads", 12))
        self.new_qformer_cross_attention_freq = max(1, int(getattr(extra_config, "new_qformer_cross_attention_freq", 2)))
        self.new_qformer_dropout = float(getattr(extra_config, "new_qformer_dropout", 0.1))
        self.new_itc_dim = int(getattr(extra_config, "new_itc_dim", 256))

        # [COPILOT] Exact adapter projections requested by user.
        self.new_vggt_in_proj = nn.Linear(self.LLM_width, self.new_qformer_encoder_width)  # 2048 -> 1408
        self.new_subtext_in_proj = nn.Linear(self.LLM_width, self.new_qformer_dim)  # 2048 -> 768

        self.new_query_tokens = nn.Parameter(torch.zeros(1, self.new_total_queries, self.new_qformer_dim))
        # [COPILOT] Learnable 2D positional embedding for the per-view 16x16 query grid.
        # Layout: [view, row, col] and then flattened in strict row-major patch order per view.
        self.new_query_pos_2d = nn.Parameter(
            torch.zeros(1, self.new_num_views, self.new_query_side, self.new_query_side, self.new_qformer_dim)
        )
        nn.init.normal_(self.new_query_tokens, std=0.02)
        nn.init.normal_(self.new_query_pos_2d, std=0.02)

        self.new_vision_proj = nn.Linear(self.new_qformer_dim, self.new_itc_dim)
        self.new_text_proj = nn.Linear(self.new_qformer_dim, self.new_itc_dim)
        self.new_itm_head = nn.Linear(self.new_qformer_dim, 2)
        self.new_temp = nn.Parameter(torch.tensor(0.07))

        self.new_qformer_pretrained_path = str(
            getattr(extra_config, "new_qformer_pretrained_path", None) or "/home/ed1116/qformer_pretrained.pth"
        )
        self.new_qformer = self._build_new_qformer_core()
        self._load_new_qformer_pretrained(self.new_qformer_pretrained_path)

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
        # [COPILOT] Avoid nested checkpoint stacks (submodule checkpoint + outer _apply_checkpoint),
        # which can trigger DDP "Expected to mark a variable ready only once" during stage transitions.
        # Keep submodule-level checkpointing off and use only the outer wrapper path.
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info(
            "Enabled gradient checkpointing for PI0Pytorch model (outer checkpoint wrapper only)"
        )

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

        # [COPILOT] Stage-1 trainables only:
        # queries + query-pos + new_vggt_in_proj + new_subtext_in_proj.
        stage1_trainable = {
            "new_query_tokens",
            "new_query_pos_2d",
            "new_vggt_in_proj.weight",
            "new_vggt_in_proj.bias",
            "new_subtext_in_proj.weight",
            "new_subtext_in_proj.bias",
        }
        if stage in (1, 3):
            for name, param in self.named_parameters():
                if name in stage1_trainable:
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

    def _build_new_qformer_core(self):
        # [COPILOT] Use custom MHA-based Q-Former core (taskaware style), with new config hyperparameters.
        return NewQFormer(
            hidden_dim=self.new_qformer_dim,
            encoder_dim=self.new_qformer_encoder_width,
            num_heads=self.new_qformer_heads,
            num_layers=self.new_qformer_layers,
            dropout=self.new_qformer_dropout,
            cross_attention_freq=self.new_qformer_cross_attention_freq,
        )

    def _load_new_qformer_pretrained(self, checkpoint_path: str) -> None:
        # [COPILOT] Load pretrained BLIP-2 Q-Former weights onto custom MHA Q-Former (shape-compatible mapping only).
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        source_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(source_state, dict):
            raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}: {type(source_state)}")

        source_qformer = {k: v for k, v in source_state.items() if k.startswith("Qformer.")}
        target_params = dict(self.new_qformer.named_parameters())
        mapped_source_keys: set[str] = set()
        loaded_target_keys: set[str] = set()

        def _copy_direct(source_key: str, target_key: str) -> bool:
            source_tensor = source_state.get(source_key)
            target = target_params.get(target_key)
            if source_tensor is None or target is None or source_tensor.shape != target.shape:
                return False
            with torch.no_grad():
                target.copy_(source_tensor.to(dtype=target.dtype))
            mapped_source_keys.add(source_key)
            loaded_target_keys.add(target_key)
            return True

        # [COPILOT] Embedding LayerNorm in BLIP-2 Q-Former maps to custom input LN.
        _copy_direct("Qformer.bert.embeddings.LayerNorm.weight", "input_ln.weight")
        _copy_direct("Qformer.bert.embeddings.LayerNorm.bias", "input_ln.bias")

        for layer_idx, layer in enumerate(self.new_qformer.layers):
            prefix = f"Qformer.bert.encoder.layer.{layer_idx}."

            # ----------------------------- Self-attention -----------------------------
            q_w = source_state.get(prefix + "attention.self.query.weight")
            k_w = source_state.get(prefix + "attention.self.key.weight")
            v_w = source_state.get(prefix + "attention.self.value.weight")
            if (
                q_w is not None
                and k_w is not None
                and v_w is not None
                and layer.self_attn.in_proj_weight.shape[0] == q_w.shape[0] + k_w.shape[0] + v_w.shape[0]
            ):
                with torch.no_grad():
                    layer.self_attn.in_proj_weight.copy_(
                        torch.cat([q_w, k_w, v_w], dim=0).to(dtype=layer.self_attn.in_proj_weight.dtype)
                    )
                mapped_source_keys.update(
                    {
                        prefix + "attention.self.query.weight",
                        prefix + "attention.self.key.weight",
                        prefix + "attention.self.value.weight",
                    }
                )
                loaded_target_keys.add(f"layers.{layer_idx}.self_attn.in_proj_weight")

            q_b = source_state.get(prefix + "attention.self.query.bias")
            k_b = source_state.get(prefix + "attention.self.key.bias")
            v_b = source_state.get(prefix + "attention.self.value.bias")
            if (
                q_b is not None
                and k_b is not None
                and v_b is not None
                and layer.self_attn.in_proj_bias.shape[0] == q_b.shape[0] + k_b.shape[0] + v_b.shape[0]
            ):
                with torch.no_grad():
                    layer.self_attn.in_proj_bias.copy_(
                        torch.cat([q_b, k_b, v_b], dim=0).to(dtype=layer.self_attn.in_proj_bias.dtype)
                    )
                mapped_source_keys.update(
                    {
                        prefix + "attention.self.query.bias",
                        prefix + "attention.self.key.bias",
                        prefix + "attention.self.value.bias",
                    }
                )
                loaded_target_keys.add(f"layers.{layer_idx}.self_attn.in_proj_bias")

            _copy_direct(prefix + "attention.output.dense.weight", f"layers.{layer_idx}.self_attn.out_proj.weight")
            _copy_direct(prefix + "attention.output.dense.bias", f"layers.{layer_idx}.self_attn.out_proj.bias")
            _copy_direct(prefix + "attention.output.LayerNorm.weight", f"layers.{layer_idx}.self_attn_ln.weight")
            _copy_direct(prefix + "attention.output.LayerNorm.bias", f"layers.{layer_idx}.self_attn_ln.bias")

            # ----------------------------- Cross-attention ----------------------------
            if layer.has_cross_attention and layer.cross_attn is not None and layer.cross_attn_ln is not None:
                cross_prefix = prefix + "crossattention."
                cross_attn = layer.cross_attn

                if hasattr(cross_attn, "q_proj_weight") and cross_attn.q_proj_weight is not None:
                    _copy_direct(cross_prefix + "self.query.weight", f"layers.{layer_idx}.cross_attn.q_proj_weight")
                    _copy_direct(cross_prefix + "self.key.weight", f"layers.{layer_idx}.cross_attn.k_proj_weight")
                    _copy_direct(cross_prefix + "self.value.weight", f"layers.{layer_idx}.cross_attn.v_proj_weight")
                elif hasattr(cross_attn, "in_proj_weight") and cross_attn.in_proj_weight is not None:
                    cq_w = source_state.get(cross_prefix + "self.query.weight")
                    ck_w = source_state.get(cross_prefix + "self.key.weight")
                    cv_w = source_state.get(cross_prefix + "self.value.weight")
                    if (
                        cq_w is not None
                        and ck_w is not None
                        and cv_w is not None
                        and cross_attn.in_proj_weight.shape[0] == cq_w.shape[0] + ck_w.shape[0] + cv_w.shape[0]
                    ):
                        with torch.no_grad():
                            cross_attn.in_proj_weight.copy_(
                                torch.cat([cq_w, ck_w, cv_w], dim=0).to(dtype=cross_attn.in_proj_weight.dtype)
                            )
                        mapped_source_keys.update(
                            {
                                cross_prefix + "self.query.weight",
                                cross_prefix + "self.key.weight",
                                cross_prefix + "self.value.weight",
                            }
                        )
                        loaded_target_keys.add(f"layers.{layer_idx}.cross_attn.in_proj_weight")

                if hasattr(cross_attn, "in_proj_bias") and cross_attn.in_proj_bias is not None:
                    cq_b = source_state.get(cross_prefix + "self.query.bias")
                    ck_b = source_state.get(cross_prefix + "self.key.bias")
                    cv_b = source_state.get(cross_prefix + "self.value.bias")
                    if (
                        cq_b is not None
                        and ck_b is not None
                        and cv_b is not None
                        and cross_attn.in_proj_bias.shape[0] == cq_b.shape[0] + ck_b.shape[0] + cv_b.shape[0]
                    ):
                        with torch.no_grad():
                            cross_attn.in_proj_bias.copy_(
                                torch.cat([cq_b, ck_b, cv_b], dim=0).to(dtype=cross_attn.in_proj_bias.dtype)
                            )
                        mapped_source_keys.update(
                            {
                                cross_prefix + "self.query.bias",
                                cross_prefix + "self.key.bias",
                                cross_prefix + "self.value.bias",
                            }
                        )
                        loaded_target_keys.add(f"layers.{layer_idx}.cross_attn.in_proj_bias")

                _copy_direct(cross_prefix + "output.dense.weight", f"layers.{layer_idx}.cross_attn.out_proj.weight")
                _copy_direct(cross_prefix + "output.dense.bias", f"layers.{layer_idx}.cross_attn.out_proj.bias")
                _copy_direct(cross_prefix + "output.LayerNorm.weight", f"layers.{layer_idx}.cross_attn_ln.weight")
                _copy_direct(cross_prefix + "output.LayerNorm.bias", f"layers.{layer_idx}.cross_attn_ln.bias")

            # ------------------------------- Query FFN --------------------------------
            _copy_direct(prefix + "intermediate_query.dense.weight", f"layers.{layer_idx}.query_fc1.weight")
            _copy_direct(prefix + "intermediate_query.dense.bias", f"layers.{layer_idx}.query_fc1.bias")
            _copy_direct(prefix + "output_query.dense.weight", f"layers.{layer_idx}.query_fc2.weight")
            _copy_direct(prefix + "output_query.dense.bias", f"layers.{layer_idx}.query_fc2.bias")
            _copy_direct(prefix + "output_query.LayerNorm.weight", f"layers.{layer_idx}.query_ffn_ln.weight")
            _copy_direct(prefix + "output_query.LayerNorm.bias", f"layers.{layer_idx}.query_ffn_ln.bias")

            # -------------------------------- Text FFN --------------------------------
            _copy_direct(prefix + "intermediate.dense.weight", f"layers.{layer_idx}.text_fc1.weight")
            _copy_direct(prefix + "intermediate.dense.bias", f"layers.{layer_idx}.text_fc1.bias")
            _copy_direct(prefix + "output.dense.weight", f"layers.{layer_idx}.text_fc2.weight")
            _copy_direct(prefix + "output.dense.bias", f"layers.{layer_idx}.text_fc2.bias")
            _copy_direct(prefix + "output.LayerNorm.weight", f"layers.{layer_idx}.text_ffn_ln.weight")
            _copy_direct(prefix + "output.LayerNorm.bias", f"layers.{layer_idx}.text_ffn_ln.bias")

        source_key_count = len(source_qformer)
        mapped_qformer = len(mapped_source_keys)
        missing_qformer = len(set(target_params.keys()) - loaded_target_keys)
        unexpected_qformer = source_key_count - mapped_qformer

        new_scope_map = {
            "vision_proj.weight": self.new_vision_proj.weight,
            "vision_proj.bias": self.new_vision_proj.bias,
            "text_proj.weight": self.new_text_proj.weight,
            "text_proj.bias": self.new_text_proj.bias,
            "itm_head.weight": self.new_itm_head.weight,
            "itm_head.bias": self.new_itm_head.bias,
            "temp": self.new_temp,
        }
        mapped_new = 0
        missing_new = 0
        for source_key, target in new_scope_map.items():
            source_tensor = source_state.get(source_key)
            if source_tensor is None or source_tensor.shape != target.shape:
                missing_new += 1
                continue
            with torch.no_grad():
                target.copy_(source_tensor.to(dtype=target.dtype))
            mapped_new += 1

        logging.info(
            "[COPILOT] new BLIP-2 preload(Q-Former): source_key_count=%d mapped=%d missing=%d unexpected=%d",
            source_key_count,
            mapped_qformer,
            missing_qformer,
            unexpected_qformer,
        )
        logging.info(
            "[COPILOT] new BLIP-2 preload(new-scope): source_key_count=%d mapped=%d missing=%d unexpected=%d",
            len(new_scope_map),
            mapped_new,
            missing_new,
            0,
        )

    def _new_qformer_encode(
        self,
        query_embeds: torch.Tensor,
        text_embeds: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [COPILOT] Q-Former forward over [queries; SUBTEXT] with cross-attn to VGGT K/V.
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=torch.bool, device=encoder_hidden_states.device
            )
        if text_embeds is not None and text_embeds.shape[1] > 0 and text_mask is None:
            text_mask = torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device)
        return self.new_qformer(
            query_tokens=query_embeds,
            text_tokens=text_embeds,
            text_mask=text_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

    def _new_query_grid(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # [COPILOT] Preserve strict [view, row, col] ordering to align query positions with SUBIMAGE patch order.
        query_tokens = self.new_query_tokens.reshape(
            1, self.new_num_views, self.new_query_side, self.new_query_side, self.new_qformer_dim
        )
        query_grid = query_tokens + self.new_query_pos_2d
        query_grid = query_grid.reshape(1, self.new_total_queries, self.new_qformer_dim)
        return query_grid.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def _new_encode_multiview_queries(
        self,
        vggt_tokens_1408: torch.Tensor,
        subtext_768: torch.Tensor | None,
        subtext_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # [COPILOT] Preserve strict query patch order: each view consumes its own contiguous 16x16 query block.
        bsz, num_views, _, _ = vggt_tokens_1408.shape
        if num_views != self.new_num_views:
            raise ValueError(f"Expected {self.new_num_views} views, got {num_views}")

        query_grid = self._new_query_grid(bsz, vggt_tokens_1408.device, vggt_tokens_1408.dtype)
        outputs = []
        for view_idx in range(num_views):
            start = view_idx * self.new_queries_per_view
            end = start + self.new_queries_per_view
            outputs.append(
                self._new_qformer_encode(
                    query_embeds=query_grid[:, start:end, :],
                    text_embeds=subtext_768,
                    text_mask=subtext_mask,
                    encoder_hidden_states=vggt_tokens_1408[:, view_idx, :, :],
                )
            )
        return torch.cat(outputs, dim=1)

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

    def _extract_prefix_hidden(self, all_hidden_states, layer_idx: int) -> torch.Tensor:
        hidden_at_layer = all_hidden_states[layer_idx]
        if isinstance(hidden_at_layer, (list, tuple)):
            return hidden_at_layer[0]
        return hidden_at_layer

    def _prepare_new_subtext(
        self,
        all_hidden_states,
        img_len: int,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # [COPILOT] SUBTEXT comes from VLA hidden layer 11 by default.
        prefix_hidden = self._extract_prefix_hidden(all_hidden_states, self.new_subtext_layer)
        subtext = prefix_hidden[:, img_len:, :]
        subtext = subtext[:, : lang_masks.shape[1], :]
        subtext_mask = lang_masks.bool()
        subtext = self.new_subtext_in_proj(subtext)
        return subtext, subtext_mask

    def _prepare_new_subimage(self, all_hidden_states, img_len: int) -> torch.Tensor:
        # [COPILOT] SUBIMAGE comes from VLA hidden layer 12 by default.
        prefix_hidden = self._extract_prefix_hidden(all_hidden_states, self.new_subimage_layer)
        return prefix_hidden[:, :img_len, :]

    def _new_pool_subtext(self, subtext_768: torch.Tensor | None, subtext_mask: torch.Tensor | None) -> torch.Tensor:
        if subtext_768 is None or subtext_768.shape[1] == 0:
            return self.new_query_tokens.new_zeros((1, self.new_qformer_dim))
        if subtext_mask is None:
            valid = torch.ones(subtext_768.shape[:2], dtype=torch.bool, device=subtext_768.device)
        else:
            valid = subtext_mask.bool()
        valid_f = valid.to(dtype=subtext_768.dtype).unsqueeze(-1)
        return (subtext_768 * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)

    def _new_itc_itm_losses(
        self,
        qformer_queries: torch.Tensor,
        subtext_768: torch.Tensor | None,
        subtext_mask: torch.Tensor | None,
        vggt_tokens_1408: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [COPILOT] BLIP-2-like ITC + ITM only. No LM/ITG objective.
        bsz = qformer_queries.shape[0]
        if bsz == 0:
            zero = qformer_queries.new_zeros(())
            return zero, zero

        pooled_subtext = self._new_pool_subtext(subtext_768, subtext_mask)
        if pooled_subtext.shape[0] != bsz:
            pooled_subtext = pooled_subtext.expand(bsz, -1)

        image_feats = F.normalize(self.new_vision_proj(qformer_queries), dim=-1)
        text_feat = F.normalize(self.new_text_proj(pooled_subtext), dim=-1)
        temp = self.new_temp.clamp(min=1e-3, max=1.0)

        sim_q2t = torch.matmul(image_feats, text_feat.transpose(0, 1))
        sim_i2t = sim_q2t.max(dim=1).values / temp
        sim_t2q = torch.einsum("bd,cqd->bcq", text_feat, image_feats)
        sim_t2i = sim_t2q.max(dim=-1).values / temp

        targets = torch.arange(bsz, device=qformer_queries.device, dtype=torch.long)
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        if bsz < 2 or subtext_768 is None:
            return loss_itc, qformer_queries.new_zeros(())

        with torch.no_grad():
            weights_i2t = sim_i2t.detach().clone()
            weights_t2i = sim_t2i.detach().clone()
            weights_i2t.fill_diagonal_(-10000.0)
            weights_t2i.fill_diagonal_(-10000.0)
            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)
            neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze(1)
            neg_img_idx = torch.multinomial(weights_t2i, 1).squeeze(1)

        subtext_neg = subtext_768[neg_text_idx]
        subtext_mask_neg = subtext_mask[neg_text_idx] if subtext_mask is not None else None
        subtext_all = torch.cat([subtext_768, subtext_768, subtext_neg], dim=0)  # pos, pos, neg_text
        if subtext_mask is not None:
            subtext_mask_all = torch.cat([subtext_mask, subtext_mask, subtext_mask_neg], dim=0)
        else:
            subtext_mask_all = None

        vggt_all = torch.cat(
            [vggt_tokens_1408, vggt_tokens_1408[neg_img_idx], vggt_tokens_1408], dim=0
        )  # pos, neg_img, pos

        itm_queries = self._new_encode_multiview_queries(vggt_all, subtext_all, subtext_mask_all)
        itm_logits = self.new_itm_head(itm_queries).mean(dim=1)
        itm_labels = torch.cat(
            [
                torch.ones(bsz, dtype=torch.long, device=qformer_queries.device),
                torch.zeros(2 * bsz, dtype=torch.long, device=qformer_queries.device),
            ],
            dim=0,
        )
        loss_itm = F.cross_entropy(itm_logits, itm_labels)
        return loss_itc, loss_itm

    def forward(
        self,
        observation,
        actions,
        vggt,
        align_proj,
        noise=None,
        time=None,
        compute_task_loss=True,
        compute_action_loss=True,
        compute_align_loss=True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Do a full training forward pass and return (action_loss, align_loss, new_aux_loss)."""
        images, img_wo_aug, img_padding_mask, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True, get_wo_aug=True
        )
        img_resize_wo_aug = preprocess_images_from_openpi(img_wo_aug)  # specific for VGGT with 518px input

        # [COPILOT] Check once whether any new_ param is trainable (True in stage 1/3, False in stage 2).
        new_needs_grad = any(
            p.requires_grad for n, p in self.named_parameters() if n.startswith("new_")
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        action_loss = actions.new_zeros(())
        align_loss = actions.new_zeros(())
        subimage_hidden = None

        if compute_action_loss or compute_align_loss:
            # =================================== VLA action/align path ===================================
            # [COPILOT] Stage 2/3: keep the original full PI0 forward path when action or alignment is needed.
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

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

            if compute_action_loss:
                suffix_out = suffix_out[:, -self.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)

                # Apply gradient checkpointing to final action projection if enabled
                def action_out_proj_func(suffix_out):
                    return self.action_out_proj(suffix_out)

                v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
                action_loss = F.mse_loss(u_t, v_t)

            if compute_align_loss:
                subimage_hidden = self._prepare_new_subimage(all_hidden_states, img_len)
        else:
            # =================================== Stage 1 fast path ===================================
            # [COPILOT] Stage 1 only trains new-query/new-adapter blocks. Skip suffix/action/align computation
            # and run prefix-only VLM in no_grad to build text conditioning tokens for Q-Former.
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            with torch.no_grad():
                prefix_outputs = self.paligemma_with_expert.paligemma.language_model.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=prefix_embs,
                    use_cache=False,
                    adarms_cond=None,
                    output_hidden_states=True,
                )
            all_hidden_states = prefix_outputs.hidden_states

        # [COPILOT] SUBTEXT from VLA hidden states (default layer 11), projected to 768.
        subtext_768, subtext_mask = self._prepare_new_subtext(all_hidden_states, img_len, lang_masks)

        # [COPILOT] VGGT hidden states (37x37 tokens per view) used as key/value for Q-Former cross-attention.
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            vggt_output = vggt(img_resize_wo_aug)
        agg_vggt_hidden = vggt_output["features"][self.vggt_layers_align]  # 24 total VGGT layers
        patch_start_idx = vggt_output["patch_start_idx"]
        vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]
        vggt_hidden_1408 = self.new_vggt_in_proj(vggt_hidden)

        # [COPILOT] Replace bilinear pooling with BLIP-2-style Q-Former compression into 16x16=256 queries per view.
        # [COPILOT] When Q-Former is frozen (stage 2), skip activation storage for backward by running under no_grad.
        if new_needs_grad:
            new_hidden = self._new_encode_multiview_queries(vggt_hidden_1408, subtext_768, subtext_mask)
        else:
            with torch.no_grad():
                new_hidden = self._new_encode_multiview_queries(vggt_hidden_1408, subtext_768, subtext_mask)
            new_hidden = new_hidden.detach()

        if compute_align_loss:
            # [COPILOT] Align Q-Former query outputs against SUBIMAGE tokens.
            if subimage_hidden is None:
                raise RuntimeError("subimage_hidden must be available when compute_align_loss=True.")
            if new_hidden.shape[:2] != subimage_hidden.shape[:2]:
                raise ValueError(
                    f"Q-Former hidden shape {new_hidden.shape} does not match SUBIMAGE hidden {subimage_hidden.shape}."
                )

            # [COPILOT] Empty-image feature masks for alignment loss.
            tokens_per_img = new_hidden.shape[1] // len(images)
            img_masks_stack = torch.stack(img_masks, dim=1)
            align_mask = torch.repeat_interleave(img_masks_stack, repeats=tokens_per_img, dim=1)

            # [COPILOT] Remove non-rectangular padded regions from the alignment loss.
            img_padding_mask = torch.stack(img_padding_mask, dim=1)
            target_size = img_padding_mask.shape[-1] // 14  # 224/14, where 14 is the patch size of Gemma encoder
            mask_downsampled = F.interpolate(
                img_padding_mask.float(),
                size=(target_size, target_size),
                mode="nearest",
            ).bool().flatten(start_dim=1)
            assert align_mask.shape == mask_downsampled.shape, \
                "align_mask shape don't match img_padding_mask shape, please manually modify the patch size of Gemma encoder (now is 14)"
            align_mask = mask_downsampled & align_mask

            # [COPILOT] Project SUBIMAGE(2048->768 via align projector) and compute cosine loss vs Q-Former outputs.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                align_loss = align_proj(subimage_hidden, new_hidden, align_mask)

        # [COPILOT][EXCLUDE] LM/ITG objective disabled; use only ITC + ITM.
        # [COPILOT] Skip ITC/ITM when task loss is not needed (stage 2) to save memory.
        if compute_task_loss:
            loss_itc, loss_itm = self._new_itc_itm_losses(new_hidden, subtext_768, subtext_mask, vggt_hidden_1408)
            new_aux_loss = loss_itc + loss_itm
        else:
            new_aux_loss = actions.new_zeros(())

        return action_loss, align_loss, new_aux_loss

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
