"""Low-rank adapters on the fused query-key-value projections of the backbone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from alhena._constants import LORA_ALPHA, LORA_RANK
from alhena._errors import BundleError

ADAPTER_NAMES: tuple[str, ...] = (
    "adapter_Q_downsample",
    "adapter_Q_upsample",
    "adapter_V_downsample",
    "adapter_V_upsample",
)
"""Names of the four tensors one adapter carries, as the bundle stores them."""

_BLOCK_PREFIX = "blocks."


class LoRAQKV(nn.Module):
    """Low-rank adaptation of a fused query-key-value projection.

    The adapters add a low-rank update to the query and value thirds of the output, and none to the
    key third. Every parameter starts at zero, so the module reproduces the wrapped layer until
    weights are loaded.

    Args:
        linear_layer: The fused `qkv` layer of a transformer block, whose output is three times as
            wide as its input.
        rank: Rank of the update.
        alpha: Scaling of the update, which is multiplied by `alpha / rank`.

    Raises:
        BundleError: If the layer is not a fused `qkv` linear layer, or `rank` is below one.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
    ) -> None:
        super().__init__()
        if not isinstance(linear_layer, nn.Linear):
            raise BundleError(
                "Adapters wrap the fused qkv linear layer of a block, got "
                f"{type(linear_layer).__name__}."
            )
        in_dim = int(linear_layer.in_features)
        out_dim = int(linear_layer.out_features)
        if out_dim != 3 * in_dim:
            raise BundleError(
                f"Expected a fused qkv layer three times as wide as its input, got {in_dim} in "
                f"and {out_dim} out."
            )
        if rank < 1:
            raise BundleError(f"Adapter rank must be at least one, got {rank}.")

        self.linear_layer = linear_layer
        self.in_dim = in_dim
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.adapter_Q_downsample = nn.Parameter(torch.zeros(in_dim, self.rank))
        self.adapter_Q_upsample = nn.Parameter(torch.zeros(self.rank, in_dim))
        self.adapter_V_downsample = nn.Parameter(torch.zeros(in_dim, self.rank))
        self.adapter_V_upsample = nn.Parameter(torch.zeros(self.rank, in_dim))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return the update the adapters add to the wrapped layer's output.

        Args:
            x: Input of the wrapped layer, `(..., in)`.

        Returns:
            The update, `(..., 3 * in)`, zero over the key third.
        """
        query = self.scale * (x @ self.adapter_Q_downsample @ self.adapter_Q_upsample)
        value = self.scale * (x @ self.adapter_V_downsample @ self.adapter_V_upsample)
        return torch.cat([query, torch.zeros_like(value), value], dim=-1)

    def delta_weight(self) -> torch.Tensor:
        """Return the update the adapters add to the fused weight, shape (3 * in, in)."""
        query = self.scale * (self.adapter_Q_downsample @ self.adapter_Q_upsample)
        value = self.scale * (self.adapter_V_downsample @ self.adapter_V_upsample)
        return torch.cat([query, torch.zeros_like(query), value], dim=1).T.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped layer and add the adapters' update.

        Args:
            x: Input of the wrapped layer, `(..., in)`.

        Returns:
            The fused query, key and value, `(..., 3 * in)`.
        """
        return self.linear_layer(x) + self.delta(x)


def adapter_key(block_index: int, name: str) -> str:
    """Return the key one adapter tensor is stored under, relative to the transformer.

    Args:
        block_index: Index of the transformer block.
        name: Name of the tensor, one of `ADAPTER_NAMES`.

    Returns:
        The key, such as `blocks.0.attn.qkv.adapter_Q_downsample`.
    """
    return f"{_BLOCK_PREFIX}{block_index}.attn.qkv.{name}"


def _blocks(vit: nn.Module) -> Sequence[nn.Module]:
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        raise BundleError(
            f"Expected a vision transformer exposing `blocks`, got {type(vit).__name__}."
        )
    return blocks


def _adapters(vit: nn.Module) -> list[LoRAQKV]:
    found: list[LoRAQKV] = []
    for index, block in enumerate(_blocks(vit)):
        layer = block.attn.qkv
        if not isinstance(layer, LoRAQKV):
            raise BundleError(
                f"Block {index} carries no adapters. Insert them before loading their weights "
                "or merging them."
            )
        found.append(layer)
    return found


def insert_adapters(vit: nn.Module, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> int:
    """Wrap the fused qkv projection of every block in a low-rank adapter.

    Args:
        vit: A vision transformer exposing `blocks`.
        rank: Rank of the adapters.
        alpha: Scaling numerator; the update is scaled by `alpha / rank`.

    Returns:
        The number of blocks adapted.

    Raises:
        BundleError: If the model exposes no blocks, or a block is already adapted.
    """
    adapted = 0
    for index, block in enumerate(_blocks(vit)):
        layer = block.attn.qkv
        if isinstance(layer, LoRAQKV):
            raise BundleError(f"Block {index} already carries adapters.")
        block.attn.qkv = LoRAQKV(layer, rank=rank, alpha=alpha)
        adapted += 1
    return adapted


def load_adapter_weights(vit: nn.Module, tensors: Mapping[str, torch.Tensor]) -> None:
    """Copy adapter weights into the adapters inserted in a transformer.

    Keys are matched from `blocks.` onward, so the prefixes of a training checkpoint are accepted.
    Every adapted block must receive its four tensors and nothing else.

    Args:
        vit: A transformer whose blocks already carry adapters.
        tensors: Adapter weights, keyed as in the bundle.

    Raises:
        BundleError: If the tensors do not match the adapted blocks, in names or in shapes.
    """
    adapters = _adapters(vit)

    supplied: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        position = key.rfind(_BLOCK_PREFIX)
        if position < 0:
            raise BundleError(
                f"Adapter tensor {key!r} names no transformer block. Keys are expected to end in "
                f"`{_BLOCK_PREFIX}<index>.attn.qkv.<adapter>`."
            )
        normalized = key[position:]
        if normalized in supplied:
            raise BundleError(
                f"Two keys name the same adapter tensor {normalized!r}. Supply one set of adapter "
                "weights, under a single prefix."
            )
        supplied[normalized] = tensor

    expected = {adapter_key(i, name) for i in range(len(adapters)) for name in ADAPTER_NAMES}
    missing = sorted(expected - set(supplied))
    unexpected = sorted(set(supplied) - expected)
    if missing or unexpected:
        raise BundleError(
            f"Adapter weights do not match the {len(adapters)} adapted blocks: {len(missing)} "
            f"missing and {len(unexpected)} unexpected tensors. First missing: {missing[:3]}. "
            f"First unexpected: {unexpected[:3]}."
        )

    with torch.no_grad():
        for index, adapter in enumerate(adapters):
            for name in ADAPTER_NAMES:
                target = getattr(adapter, name)
                tensor = supplied[adapter_key(index, name)]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise BundleError(
                        f"Adapter tensor {adapter_key(index, name)!r} has shape "
                        f"{tuple(tensor.shape)} where rank {adapter.rank} needs "
                        f"{tuple(target.shape)}."
                    )
                target.copy_(tensor.to(dtype=torch.float32))


def merge_adapters(vit: nn.Module) -> None:
    """Fold the inserted adapters into the fused weights and restore plain linear layers.

    The merged and inserted forms agree to float32 rounding: on the released bundle, at most 2.3e-05
    in absolute terms and 2.9e-06 in relative terms.

    Args:
        vit: A transformer whose blocks carry adapters holding their final weights.

    Raises:
        BundleError: If a block carries no adapter.
    """
    for block, adapter in zip(_blocks(vit), _adapters(vit), strict=True):
        linear = adapter.linear_layer
        with torch.no_grad():
            linear.weight.add_(adapter.delta_weight().to(dtype=linear.weight.dtype))
        block.attn.qkv = linear
