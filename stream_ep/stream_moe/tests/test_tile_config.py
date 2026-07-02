"""Tests for the public TileConfig surface.

Covers the three pieces that make up the ``tile_config`` API:
  - ``default_tile_config(I, H)`` — the bench-tuned resolved baseline, including
    the power-of-2 divisibility fallback when a default tile_n doesn't divide.
  - ``TileConfig.validate(I, H)`` — the four StreamEP-specific output-tile_n
    divisibility constraints (None fields skipped).
  - ``_resolve_tile_config`` — overlay a caller's non-None overrides onto the
    baseline. The load-bearing property: pinning ONE knob must NOT freeze the
    others at a hardcoded default — they stay auto-picked for the shape.

Pure tile-math (no kernels launch), but the import pulls ``stream_ep`` (CUDA
extension), so this collects on a GPU node like the other stream_moe pytest
modules.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from stream_ep.stream_moe.stream_moe import (
    TileConfig,
    _resolve_tile_config,
    default_tile_config,
)

# (I, H): first row hits all four bench-tuned defaults; the rest force a
# power-of-2 fallback on one or more tile_n values.
_SHAPES = [
    (1536, 2048),
    (2048, 2048),
    (2048, 1536),
    (768, 1024),
    (1024, 768),
    (2560, 2560),
    (512, 384),
    (320, 512),
]


@pytest.mark.parametrize("I, H", _SHAPES)
def test_default_config_self_consistent(I, H):
    cfg = default_tile_config(I, H)
    # The picker is responsible for emitting a valid config — must not raise.
    cfg.validate(I, H)
    # Restate the four constraints independently of validate().
    assert (2 * I) % cfg.tile_n_a == 0 and I % cfg.tile_n_a == 0
    assert H % cfg.tile_n_y == 0
    assert I % cfg.tile_n_y_bwd == 0
    assert H % cfg.tile_n_a_bwd == 0
    assert cfg.tile_m == 128


def test_bench_tuned_defaults_when_divisible():
    # I=1536, H=2048: 2I=3072 and I=1536 both divisible by 192, H by 256,
    # I by 192, H by 256 -> all four hit their bench-tuned defaults.
    cfg = default_tile_config(1536, 2048)
    assert (cfg.tile_n_a, cfg.tile_n_y, cfg.tile_n_y_bwd, cfg.tile_n_a_bwd) == (
        192,
        256,
        192,
        256,
    )


def test_fallback_when_default_does_not_divide():
    # H=384 is not divisible by the tile_n_y default (256); picker must fall
    # back to a power-of-2 that divides it. A frozen 256 would be invalid here.
    cfg = default_tile_config(512, 384)
    assert cfg.tile_n_y != 256
    assert 384 % cfg.tile_n_y == 0
    assert cfg.tile_n_y & (cfg.tile_n_y - 1) == 0  # power of two


@pytest.mark.parametrize(
    "field",
    ["tile_n_a", "tile_n_y", "tile_n_y_bwd", "tile_n_a_bwd"],
)
def test_validate_rejects_bad_tile_n(field):
    # 7 divides none of 2I / I / H for this shape.
    cfg = TileConfig(**{field: 7})
    with pytest.raises(ValueError, match=field):
        cfg.validate(1536, 2048)


def test_validate_skips_none_fields():
    # An all-None config (and a partial one with only non-tile_n knobs set)
    # has nothing concrete to check, so validate must not raise / TypeError.
    TileConfig().validate(1536, 2048)
    TileConfig(num_sms_a=120, swizzle_dW1=4).validate(1536, 2048)


def test_resolve_none_is_bare_baseline():
    assert _resolve_tile_config(None, 2048, 2048) == default_tile_config(2048, 2048)
    assert _resolve_tile_config(TileConfig(), 2048, 2048) == default_tile_config(
        2048, 2048
    )


def test_resolve_pins_override_keeps_rest():
    I, H = 2048, 2048
    base = default_tile_config(I, H)
    resolved = _resolve_tile_config(TileConfig(tile_n_a=128, num_sms_a=120), I, H)
    # Pinned fields take the override.
    assert resolved.tile_n_a == 128
    assert resolved.num_sms_a == 120
    # Everything else stays exactly the shape-picked baseline.
    assert resolved == replace(base, tile_n_a=128, num_sms_a=120)


def test_resolve_unset_fields_stay_autopicked_not_frozen():
    # The load-bearing property. On a shape where the bench-tuned tile_n_a
    # default (192) does NOT divide 2I, pinning an UNRELATED knob must leave
    # tile_n_a auto-picked (256 here), never frozen at 192.
    I, H = 2048, 2048  # 2I=4096 -> 192 doesn't divide; picker -> 256
    assert default_tile_config(I, H).tile_n_a == 256
    resolved = _resolve_tile_config(TileConfig(num_sms_a=64), I, H)
    assert resolved.tile_n_a == 256  # auto-picked for the shape, not 192
    resolved.validate(I, H)  # and therefore valid


def test_resolve_validates_bad_override():
    with pytest.raises(ValueError, match="tile_n_y"):
        _resolve_tile_config(TileConfig(tile_n_y=7), 1536, 2048)


# ── tile_m backward-codegen guard (atom_layout_n=2) ──────────────────────────
# tile_m 192/320 force quack atom_layout_n=2, which the streaming backward GEMMs
# can't codegen. 320 always; 192 only when a tile_n > 128. I=H=2048 so 128 and
# 256 both divide every dim — isolates the tile_m check from divisibility.

def test_validate_rejects_tile_m_320_even_with_small_tile_n():
    cfg = TileConfig(
        tile_m=320, tile_n_a=128, tile_n_y=128, tile_n_y_bwd=128,
        tile_n_a_bwd=128, tile_n_dW1=128, tile_n_dW2=128,
    )
    with pytest.raises(ValueError, match="tile_m=320"):
        cfg.validate(2048, 2048)


def test_validate_rejects_tile_m_192_with_large_tile_n():
    # tile_n_y_bwd=256 divides I=2048 (so divisibility is fine) but >128, which
    # the tile_m=192 guard must reject and name.
    cfg = TileConfig(tile_m=192, tile_n_y_bwd=256)
    with pytest.raises(ValueError, match="tile_m=192.*tile_n_y_bwd=256"):
        cfg.validate(2048, 2048)


def test_validate_accepts_tile_m_192_with_all_tile_n_le_128():
    # The verified-working config: tile_m=192 + every tile_n <= 128.
    TileConfig(
        tile_m=192, tile_n_a=128, tile_n_y=128, tile_n_y_bwd=128,
        tile_n_a_bwd=128, tile_n_dW1=128, tile_n_dW2=128,
    ).validate(2048, 2048)


def test_validate_accepts_tile_m_256_with_large_tile_n():
    # 256 keeps atom_layout_n=1, so it is NOT gated on tile_n (unlike 192/320).
    TileConfig(tile_m=256, tile_n_y_bwd=256).validate(2048, 2048)


def test_resolve_rejects_tile_m_192_with_autopicked_tile_n():
    # Resolved config: tile_m=192 override + auto tile_n (256 at this shape) ->
    # validate must reject, since auto tile_n exceed 128.
    with pytest.raises(ValueError, match="tile_m=192"):
        _resolve_tile_config(TileConfig(tile_m=192), 2048, 2048)
