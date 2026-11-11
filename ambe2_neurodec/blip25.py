"""ctypes binding of the bundled ambe2-c AMBE+2 decoder (csrc/, MIT; a C port of blip25-vocoder).

The neural decoder uses this reference decoder's output as one of its inputs, so the library is needed at run time.
It is compiled on first use with the C compiler in $CC (or cc / gcc / clang on PATH) into ambe2_neurodec/_lib/.
"""
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
LIBDIR = HERE / "_lib"
SUFFIX = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
LIB = LIBDIR / ("blip25" + SUFFIX)
DEC_SIZE = 4096  # >= sizeof(blip25_decoder) (2772 bytes)
FRAME = 160


def build(cc: str | None = None, force: bool = False) -> Path:
    """Compile csrc/ into a shared library (no-op if it exists, unless force)."""
    if LIB.exists() and not force:
        return LIB
    cc = cc or os.environ.get("CC") or next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc is None:
        raise RuntimeError("no C compiler found: set $CC, or install gcc / clang (on Windows e.g. MinGW-w64)")
    LIBDIR.mkdir(exist_ok=True)
    srcs = sorted(str(p) for p in (CSRC / "src").glob("*.c"))
    cmd = [cc, "-O2", "-std=c99", "-shared", "-fPIC", f"-I{CSRC / 'include'}", f"-I{CSRC / 'src'}", *srcs,
           "-o", str(LIB)]
    if sys.platform == "win32":
        cmd.remove("-fPIC")
    else:
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return LIB


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(build()))
        P, I = ctypes.c_void_p, ctypes.c_int
        _lib.blip25_init.argtypes = [P]
        _lib.blip25_decode_bits.argtypes = [P, P, I, P]
        _lib.blip25_decode_lost.argtypes = [P, P]
        _lib.blip25_thumbdv_to_bits.argtypes = [P, P]
        _lib.blip25_fec_3600.argtypes = [P, P, P]
    return _lib


def thumbdv_to_bits(frames: np.ndarray) -> np.ndarray:
    """uint8 [T, 7] AMBE-3000R rate-34 frames -> uint8 [T, 49] bits in ambe_d order."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    out = np.zeros((len(frames), 49), np.uint8)
    for i in range(len(frames)):
        lib().blip25_thumbdv_to_bits(frames[i].ctypes.data, out[i].ctypes.data)
    return out


def fec_3600(frames: np.ndarray):
    """uint8 [T, 9] rate-33 (3600x2450) frames -> bits [T, 49], lost [T] (uncorrectable leading codeword),
    corrected-error counts [T]."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    bits = np.zeros((len(frames), 49), np.uint8)
    lost = np.zeros(len(frames), bool)
    nerr = np.zeros(len(frames), np.int32)
    for i in range(len(frames)):
        r = lib().blip25_fec_3600(frames[i].ctypes.data, bits[i].ctypes.data, None)
        lost[i] = r < 0
        nerr[i] = max(r, 0)
    return bits, lost, nerr


def decode(bits: np.ndarray, lost: np.ndarray | None = None) -> np.ndarray:
    """The reference AMBE+2 decoder: bits [T, 49] (+ lost mask [T]) -> int16 [160 T] at 8 kHz.
    Frames marked lost go through the decoder's concealment (repeat and fade)."""
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    mem = ctypes.create_string_buffer(DEC_SIZE)
    lib().blip25_init(mem)
    out = np.zeros(len(bits) * FRAME, np.int16)
    for i in range(len(bits)):
        if lost is not None and lost[i]:
            lib().blip25_decode_lost(mem, out[i * FRAME:].ctypes.data)
        else:
            lib().blip25_decode_bits(mem, bits[i].ctypes.data, 0, out[i * FRAME:].ctypes.data)
    return out
