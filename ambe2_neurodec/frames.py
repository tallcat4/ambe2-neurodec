"""Frame file readers -> bits uint8 [T, 49] (ambe_d order), lost bool [T].

  rate34  7 bytes per frame, AMBE-3000R / ThumbDV rate-34 order (ambe2-c's wav2ambe output)
  ambe_d  7 bytes per frame, 49 bits in ambe_d order (mbelib / DSD), MSB first, 7 pad bits
  rate33  9 bytes per frame, AMBE-3000R rate 33 (3600x2450 with FEC); an uncorrectable leading codeword = erasure
  hex     one frame per line, 14 hex digits in rate-34 order; an empty line or "-" = erased frame
"""
from pathlib import Path

import numpy as np

from . import blip25

SIZES = {"rate34": 7, "ambe_d": 7, "rate33": 9}


def _unpack_ambe_d(frames: np.ndarray) -> np.ndarray:
    return np.unpackbits(frames, axis=1)[:, :49]


def guess_format(path: str | Path) -> str:
    return "hex" if Path(path).suffix.lower() in (".hex", ".txt") else "rate34"


def read(path: str | Path, fmt: str | None = None):
    """fmt: rate34 | ambe_d | rate33 | hex (default: hex for .hex / .txt, otherwise rate34)."""
    path = Path(path)
    fmt = fmt or guess_format(path)
    if fmt == "hex":
        rows, lost = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s == "-":
                rows.append(bytes(7))
                lost.append(True)
                continue
            b = bytes.fromhex(s)
            if len(b) != 7:
                raise ValueError(f"{path}: expected 14 hex digits per line, got {s!r}")
            rows.append(b)
            lost.append(False)
        frames = np.frombuffer(b"".join(rows), np.uint8).reshape(-1, 7)
        return blip25.thumbdv_to_bits(frames), np.array(lost, bool)
    n = SIZES[fmt]
    data = np.fromfile(path, np.uint8)
    if len(data) % n:
        raise ValueError(f"{path}: {len(data)} bytes is not a multiple of the {fmt} frame size ({n})")
    frames = data.reshape(-1, n)
    if fmt == "rate34":
        return blip25.thumbdv_to_bits(frames), np.zeros(len(frames), bool)
    if fmt == "ambe_d":
        return _unpack_ambe_d(frames), np.zeros(len(frames), bool)
    bits, lost, _ = blip25.fec_3600(frames)
    return bits, lost


def write_hex(path: str | Path, frames: np.ndarray, lost: np.ndarray | None = None) -> None:
    """rate-34 frames uint8 [T, 7] (+ lost mask) -> hex text (erased frames as "-")."""
    lines = ["-" if lost is not None and lost[i] else bytes(f).hex() for i, f in enumerate(frames)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
