"""Named configuration registry for world models, with a factory function
for the diffusion decoder.

Naming convention for world configs: ``{component}-{depth}L-{size}``

    - component: ``world``
    - depth: number of hierarchical levels (e.g. ``2L``, ``3L``, ``4L``)
    - size: ``tiny``, ``small``, ``base``, ``large``

Examples::

    from worlds1k.model.configs import get_config, list_configs
    from worlds1k.model.configs import build_diffusion_decoder

    cfg = get_config("world-3L-base")
    model = WorldModel.from_config(cfg)

    decoder = build_diffusion_decoder(cfg, d_model=128, num_inference_steps=20)
"""

from __future__ import annotations

from dataclasses import asdict

from .diffusion_decoder import AdaGNDiffusionDecoder, CrossAttnDiffusionDecoder, DiffusionDecoderBase
from .world_model import WorldModelConfig

# ---------------------------------------------------------------------------
# World model presets
# ---------------------------------------------------------------------------

_WORLD_CONFIGS: dict[str, WorldModelConfig] = {
    # --- 2-level ---
    # Sizing: ~4x geometric scaling per tier
    #   tiny ~2.5M, small ~17M, base ~85M, large ~283M, xlarge ~865M
    "world-2L-tiny": WorldModelConfig(
        num_levels=2,
        d_input=512,
        d_latents=[192, 96],
        d_actions=[48, 24],
        temporal_strides=[1, 8],
        level_weights=[1.0, 1.0],
        num_transformer_heads=4,
        num_transformer_layers=3,
        backbone_name="dinov2-small",
    ),
    "world-2L-small": WorldModelConfig(
        num_levels=2,
        d_input=512,
        d_latents=[384, 192],
        d_actions=[96, 48],
        temporal_strides=[1, 8],
        level_weights=[1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=6,
        backbone_name="dinov2-small",
    ),
    "world-2L-base": WorldModelConfig(
        num_levels=2,
        d_input=768,
        d_latents=[768, 384],
        d_actions=[192, 96],
        temporal_strides=[1, 8],
        level_weights=[1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=8,
        backbone_name="dinov2-base",
    ),
    "world-2L-large": WorldModelConfig(
        num_levels=2,
        d_input=1024,
        d_latents=[1024, 512],
        d_actions=[256, 128],
        temporal_strides=[1, 8],
        level_weights=[1.0, 1.0],
        num_transformer_heads=16,
        num_transformer_layers=16,
        backbone_name="dinov2-large",
    ),
    "world-2L-xlarge": WorldModelConfig(
        num_levels=2,
        d_input=1024,
        d_latents=[2048, 1024],
        d_actions=[512, 256],
        temporal_strides=[1, 8],
        level_weights=[1.0, 1.0],
        num_transformer_heads=32,
        num_transformer_layers=12,
        backbone_name="dinov2-large",
    ),
    # --- 3-level (default) ---
    "world-3L-tiny": WorldModelConfig(
        num_levels=3,
        d_input=512,
        d_latents=[192, 96, 48],
        d_actions=[48, 24, 12],
        temporal_strides=[1, 8, 8],
        level_weights=[1.0, 1.0, 1.0],
        num_transformer_heads=4,
        num_transformer_layers=3,
        backbone_name="dinov2-small",
    ),
    "world-3L-small": WorldModelConfig(
        num_levels=3,
        d_input=512,
        d_latents=[384, 192, 96],
        d_actions=[96, 48, 24],
        temporal_strides=[1, 8, 8],
        level_weights=[1.0, 1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=6,
        backbone_name="dinov2-small",
    ),
    "world-3L-base": WorldModelConfig(
        num_levels=3,
        d_input=768,
        d_latents=[768, 384, 192],
        d_actions=[192, 96, 48],
        temporal_strides=[1, 8, 8],
        level_weights=[1.0, 1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=8,
        backbone_name="dinov2-base",
    ),
    "world-3L-large": WorldModelConfig(
        num_levels=3,
        d_input=1024,
        d_latents=[1024, 512, 256],
        d_actions=[256, 128, 64],
        temporal_strides=[1, 8, 8],
        level_weights=[1.0, 1.0, 1.0],
        num_transformer_heads=16,
        num_transformer_layers=16,
        backbone_name="dinov2-large",
    ),
    "world-3L-xlarge": WorldModelConfig(
        num_levels=3,
        d_input=1024,
        d_latents=[2048, 1024, 512],
        d_actions=[512, 256, 128],
        temporal_strides=[1, 8, 8],
        level_weights=[1.0, 1.0, 1.0],
        num_transformer_heads=32,
        num_transformer_layers=12,
        backbone_name="dinov2-large",
    ),
    # --- 4-level ---
    "world-4L-tiny": WorldModelConfig(
        num_levels=4,
        d_input=512,
        d_latents=[192, 96, 48, 24],
        d_actions=[48, 24, 12, 6],
        temporal_strides=[1, 4, 4, 4],
        level_weights=[1.0, 1.0, 1.0, 1.0],
        num_transformer_heads=4,
        num_transformer_layers=3,
        backbone_name="dinov2-small",
    ),
    "world-4L-small": WorldModelConfig(
        num_levels=4,
        d_input=512,
        d_latents=[384, 192, 96, 48],
        d_actions=[96, 48, 24, 12],
        temporal_strides=[1, 4, 4, 4],
        level_weights=[1.0, 1.0, 1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=6,
        backbone_name="dinov2-small",
    ),
    "world-4L-base": WorldModelConfig(
        num_levels=4,
        d_input=768,
        d_latents=[768, 384, 192, 96],
        d_actions=[192, 96, 48, 24],
        temporal_strides=[1, 4, 4, 4],
        level_weights=[1.0, 1.0, 1.0, 1.0],
        num_transformer_heads=8,
        num_transformer_layers=8,
        backbone_name="dinov2-base",
    ),
    "world-4L-large": WorldModelConfig(
        num_levels=4,
        d_input=1024,
        d_latents=[1024, 512, 256, 128],
        d_actions=[256, 128, 64, 32],
        temporal_strides=[1, 4, 4, 4],
        level_weights=[1.0, 1.0, 1.0, 1.0],
        num_transformer_heads=16,
        num_transformer_layers=16,
        backbone_name="dinov2-large",
    ),
    "world-4L-xlarge": WorldModelConfig(
        num_levels=4,
        d_input=1024,
        d_latents=[2048, 1024, 512, 256],
        d_actions=[512, 256, 128, 64],
        temporal_strides=[1, 4, 4, 4],
        level_weights=[1.0, 1.0, 1.0, 1.0],
        num_transformer_heads=32,
        num_transformer_layers=12,
        backbone_name="dinov2-large",
    ),
}

# ---------------------------------------------------------------------------
# Unified registry (world model configs only)
# ---------------------------------------------------------------------------

_ALL_CONFIGS: dict[str, WorldModelConfig] = dict(_WORLD_CONFIGS)


def get_config(name: str) -> WorldModelConfig:
    """Look up a named world model configuration.

    Parameters
    ----------
    name : str
        Config name, e.g. ``"world-3L-base"``.

    Returns
    -------
    WorldModelConfig
        A copy of the requested configuration.

    Raises
    ------
    KeyError
        If the name is not found in the registry.
    """
    if name not in _ALL_CONFIGS:
        available = ", ".join(sorted(_ALL_CONFIGS))
        raise KeyError(f"Unknown config {name!r}. Available: {available}")
    cfg = _ALL_CONFIGS[name]
    return WorldModelConfig(**asdict(cfg))


def list_configs(component: str | None = None) -> dict[str, WorldModelConfig]:
    """List available configurations.

    Parameters
    ----------
    component : str or None
        Filter by component prefix (e.g. ``"world"``).
        ``None`` returns all.
    """
    if component is None:
        return dict(_ALL_CONFIGS)
    return {k: v for k, v in _ALL_CONFIGS.items() if k.startswith(component)}


# ---------------------------------------------------------------------------
# Diffusion decoder factory
# ---------------------------------------------------------------------------


_DECODER_SIZE_TIERS: dict[str, dict[str, int]] = {
    "adagn-small": 32,
    "adagn-base": 64,
    "adagn-large": 128,
    "unet-small": 64,
    "unet-base": 128,
    "unet-large": 256,
}


def build_diffusion_decoder(
    world_cfg: WorldModelConfig,
    arch: str = "adagn",
    size: str = "base",
    num_inference_steps: int = 20,
    num_train_timesteps: int = 1000,
) -> DiffusionDecoderBase:
    """Build a diffusion decoder matched to a world model configuration.

    Parameters
    ----------
    world_cfg : WorldModelConfig
        The world model configuration (used for d_latent, image_size,
        image_channels).
    arch : str
        Backend: ``"adagn"`` (MPS-friendly) or ``"unet"`` (cross-attention, GPU).
    size : str
        Size tier: ``"small"``, ``"base"``, or ``"large"``.
    num_inference_steps : int
        DDIM steps for fast sampling (default 20).
    num_train_timesteps : int
        Number of diffusion timesteps (default 1000).

    Returns
    -------
    DiffusionDecoder
        A diffusion decoder instance.
    """
    key = f"{arch}-{size}"
    if key not in _DECODER_SIZE_TIERS:
        available = ", ".join(sorted(_DECODER_SIZE_TIERS))
        raise ValueError(f"Unknown decoder config {key!r}. Available: {available}")

    d_model = _DECODER_SIZE_TIERS[key]
    cls = AdaGNDiffusionDecoder if arch == "adagn" else CrossAttnDiffusionDecoder

    return cls(
        d_latent=world_cfg.d_input,
        image_size=world_cfg.image_size,
        image_channels=world_cfg.image_channels,
        d_model=d_model,
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
    )


def list_decoder_configs(world_cfg: WorldModelConfig) -> dict[str, int]:
    """Return decoder config names and their param counts for a given world model."""
    counts = {}
    for key, d_model in _DECODER_SIZE_TIERS.items():
        arch = key.rsplit("-", 1)[0]
        cls = AdaGNDiffusionDecoder if arch == "adagn" else CrossAttnDiffusionDecoder
        dec = cls(
            d_latent=world_cfg.d_input,
            image_size=world_cfg.image_size,
            image_channels=world_cfg.image_channels,
            d_model=d_model,
            num_train_timesteps=10,
        )
        counts[key] = sum(p.numel() for p in dec.parameters())
    return counts
