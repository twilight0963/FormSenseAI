"""ViTPose (simple decoder) in Keras, weight-compatible with ``usyd-community/vitpose-*-simple``.

Plain ViT backbone (pre-LN blocks, no CLS token) on a 16×12 patch grid, then a
ReLU → 4× bilinear upsample → 3×3 conv head giving stride-4 heatmaps. The layout
follows Hugging Face's ``VitPoseForPoseEstimation`` so ``scripts/convert_vitpose.py``
can copy its weights tensor by tensor.
"""

from __future__ import annotations

import keras
from keras import layers

from posenet.config import ModelConfig
from posenet.keypoints import NUM_KEYPOINTS

IMAGENET_MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
IMAGENET_STD = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
LN_EPS = 1e-12  # HF VitPoseBackboneConfig.layer_norm_eps

# name → (hidden size, depth, heads); MLP ratio is 4 for all.
VARIANTS = {
    "vitpose_s": (384, 12, 12),
    "vitpose_b": (768, 12, 12),
    "vitpose_l": (1024, 24, 16),
}


@keras.saving.register_keras_serializable(package="posenet")
class PositionEmbedding(layers.Layer):
    """Learned position embeddings for ``num_patches`` tokens plus a CLS slot.

    ViTPose drops the CLS token but adds its embedding to every patch, so the
    checkpoint's ``(1, N + 1, D)`` tensor is kept as-is.
    """

    def __init__(self, num_patches: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, self.num_patches + 1, input_shape[-1]),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="embedding",
        )

    def call(self, x):
        pos = keras.ops.cast(self.embedding, x.dtype)
        return x + pos[:, 1:] + pos[:, :1]

    def get_config(self):
        return {**super().get_config(), "num_patches": self.num_patches}


def _block(x, dim: int, heads: int, name: str):
    y = layers.LayerNormalization(epsilon=LN_EPS, name=f"{name}_ln1")(x)
    y = layers.MultiHeadAttention(heads, dim // heads, name=f"{name}_attn")(y, y)
    x = layers.Add(name=f"{name}_res1")([x, y])
    y = layers.LayerNormalization(epsilon=LN_EPS, name=f"{name}_ln2")(x)
    y = layers.Dense(4 * dim, activation="gelu", name=f"{name}_fc1")(y)  # exact (erf) GELU
    y = layers.Dense(dim, name=f"{name}_fc2")(y)
    return layers.Add(name=f"{name}_res2")([x, y])


def build_vitpose(
    cfg: ModelConfig, num_keypoints: int = NUM_KEYPOINTS, variant: str = "vitpose_b"
) -> keras.Model:
    """Input: (B, H, W, 3) RGB in [0, 255]. Output: (B, H/4, W/4, K) float32 heatmaps."""
    if cfg.heatmap_stride != 4:
        raise ValueError("The simple decoder upsamples to stride 4; heatmap_stride must be 4.")
    dim, depth, heads = VARIANTS[variant]
    h, w = cfg.input_shape
    if h % 16 or w % 16:
        raise ValueError(f"input {h}x{w} must be a multiple of the 16-pixel patch")
    gh, gw = h // 16, w // 16

    inputs = keras.Input((h, w, 3), name="image")
    x = layers.Normalization(mean=IMAGENET_MEAN, variance=[s * s for s in IMAGENET_STD], name="normalize")(
        inputs
    )
    # padding=2 as in the original ViTPose; the extra 4 pixels never form a full patch.
    x = layers.ZeroPadding2D(2, name="patch_pad")(x)
    x = layers.Conv2D(dim, 16, strides=16, name="patch_embed")(x)
    x = layers.Cropping2D(((0, x.shape[1] - gh), (0, x.shape[2] - gw)), name="patch_grid")(x)
    x = layers.Reshape((gh * gw, dim), name="tokens")(x)
    x = PositionEmbedding(gh * gw, name="pos_embed")(x)

    for i in range(depth):
        x = _block(x, dim, heads, name=f"block{i}")
    x = layers.LayerNormalization(epsilon=LN_EPS, name="final_ln")(x)

    x = layers.Reshape((gh, gw, dim), name="feature_map")(x)
    x = layers.ReLU(name="head_relu")(x)
    # float32: see the note on the decoder upsampling in posenet.model.network.
    x = layers.UpSampling2D(4, interpolation="bilinear", dtype="float32", name="head_up")(x)
    heatmaps = layers.Conv2D(num_keypoints, 3, padding="same", dtype="float32", name="heatmaps")(x)
    return keras.Model(inputs, heatmaps, name=variant)


def block_index(var_path: str, depth: int) -> int:
    """Depth bucket of a variable for layer-wise LR decay and freezing.

    0 = patch/position embedding, i + 1 = transformer block i, depth + 1 = final LN and head.
    """
    if var_path.startswith(("patch_embed", "pos_embed")):
        return 0
    if var_path.startswith("block"):
        return int(var_path[len("block") :].split("_", 1)[0]) + 1
    return depth + 1
