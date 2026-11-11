"""ambe2neurodec: AMBE+2 2450 bps frames -> 8 kHz mono 16-bit WAV (neural decoder)."""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

from . import blip25, frames
from .model import SR, decode, load

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "ambe49_causal_v1.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ambe2neurodec", description=__doc__)
    ap.add_argument("input", help="frame file")
    ap.add_argument("output", help="output WAV (8 kHz, mono, 16 bit)")
    ap.add_argument("--format", choices=["rate34", "ambe_d", "rate33", "hex"], default=None,
                    help="rate34: 7 bytes per frame, AMBE-3000R rate-34 order (default); ambe_d: 7 bytes, mbelib order; "
                         "rate33: 9 bytes with FEC; hex: 14 hex digits per line, '-' = erased (default for .hex/.txt)")
    ap.add_argument("--reference", default=None, help="also write the reference AMBE+2 decoder's output (WAV)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="the vocoder's excitation noise")
    a = ap.parse_args(argv)

    bits, lost = frames.read(a.input, a.format)
    wav = decode(load(a.model, a.device), bits, lost, a.device, seed=a.seed)
    sf.write(a.output, wav, SR, subtype="PCM_16")
    if a.reference:
        sf.write(a.reference, blip25.decode(bits, lost), SR, subtype="PCM_16")
    print(f"{len(bits)} frames ({len(bits) * 0.02:.2f} s), erased {int(lost.sum())} -> {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
