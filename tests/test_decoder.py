from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ambe2_neurodec import blip25, decode, frames, load
from ambe2_neurodec.model import bit_input

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "ambe49_causal_v1.pt"
EX = ROOT / "examples" / "qwen3-tts"


@pytest.fixture(scope="module")
def G():
    return load(MODEL)


def random_bits(n, seed=0):
    return np.random.default_rng(seed).integers(0, 2, (n, 49)).astype(np.uint8)


def frame_db(y):
    y = y[: len(y) // 160 * 160].reshape(-1, 160)
    return 10 * np.log10((y.astype(np.float64) ** 2).mean(1) + 1e-10)


def close_in_energy(a, b, mean_db=0.5):
    n = min(len(a), len(b))
    ea, eb = frame_db(a[:n]), frame_db(b[:n])
    act = ea > ea.max() - 40
    return np.abs(ea - eb)[act].mean() < mean_db


def test_reference_decoder_matches_example():
    # the bundled C decoder reproduces the stored reference decode bit for bit
    bits, lost = frames.read(EX / "female" / "frames_clean.ambe")
    ref, _ = sf.read(EX / "female" / "decoded_ambe2c_clean.wav", dtype="int16")
    assert np.array_equal(blip25.decode(bits, lost), ref)


def test_hex_erasures_roundtrip(tmp_path):
    src = EX / "male" / "frames_loss17.hex"
    bits, lost = frames.read(src)
    assert lost.any() and not lost.all()
    rate34 = np.zeros((len(bits), 7), np.uint8)
    for i, line in enumerate(src.read_text().splitlines()):
        if line != "-":
            rate34[i] = np.frombuffer(bytes.fromhex(line), np.uint8)
    p = tmp_path / "f.hex"
    frames.write_hex(p, rate34, lost)
    b2, l2 = frames.read(p)
    assert np.array_equal(l2, lost) and np.array_equal(b2[~lost], bits[~lost])


def test_decode_shape_and_range(G):
    bits = random_bits(37)
    lost = np.zeros(37, bool)
    lost[10:13] = True
    y = decode(G, bits, lost)
    assert y.shape == (37 * 160,) and np.isfinite(y).all() and np.abs(y).max() <= 1


def test_decode_is_reproducible(G):
    bits = random_bits(50, 2)
    assert np.array_equal(decode(G, bits, seed=3), decode(G, bits, seed=3))


def test_chunked_close_to_whole(G):
    bits, lost = frames.read(EX / "female" / "frames_clean.ambe")
    whole = decode(G, bits, lost)
    chunked = decode(G, bits, lost, chunk_frames=100, context=100)
    assert close_in_energy(whole, chunked)


def test_example_decode_is_reproduced(G):
    bits, lost = frames.read(EX / "male" / "frames_ber3.ambe")
    y = decode(G, bits, lost)
    stored, _ = sf.read(EX / "male" / "decoded_ambe49_causal_v1_ber3.wav", dtype="float32")
    assert close_in_energy(y, stored)  # CPU here, GPU (TF32 convolutions) for the stored file


def test_causal_model_uses_one_frame_ahead(G):
    # changing the bits of frame t leaves every sample before frame t - 1 (minus the iSTFT overlap, 256 samples)
    # unchanged. The reference decoder's output is held fixed, so this checks the network alone.
    assert G.lookahead == 1
    T, t = 120, 80
    bits = torch.from_numpy(bit_input(random_bits(T, 3), np.zeros(T, bool)))[None]
    ref = torch.from_numpy(np.random.default_rng(4).standard_normal(160 * T).astype(np.float32) * 0.05)[None]
    flipped = bits.clone()
    flipped[0, t, :49] *= -1
    torch.manual_seed(0)
    y0 = G(bits, ref)[0]
    torch.manual_seed(0)
    y1 = G(flipped, ref)[0]
    first = int(torch.nonzero((y1 - y0).abs() > 1e-6).min())
    assert 160 * (t - 1) - 256 <= first < 160 * t
