"""The neural AMBE+2 decoder: 49 bits per 20 ms frame -> 8 kHz speech.

    received bits [T, 49] (+ erasure mask) -----------------------------> bit embedding --+
             \\-> reference AMBE+2 decoder (blip25, csrc/) -> 80-band log-mel -> mel embedding --+
                                                                                               |
    FocalNet (focal modulation, as FocalCodec's decompressor) -> residual on the reference log-mel
    -> clean-speech log-mel [T, 80] -> small iSTFT (Vocos-style) vocoder pre-trained on clean speech -> 8 kHz

The vocoder is the one of https://github.com/tallcat4/neurodec (frozen during training); its excitation input is
ignored by the network, so it gets a constant (see docs/method.md).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import blip25
from .focalnet import FocalDecoder
from .istft import ISTFTGenerator, conv1d

FRAME = 160  # samples per 20 ms frame at 8 kHz
SR = 8000


def mel_filterbank(n_fft: int, n_mels: int, sr: int = SR):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = pts[i: i + 3]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l + 1e-9), (r - bins) / (r - c + 1e-9)), 0, None)
    return torch.tensor(fb, dtype=torch.float32)


class LogMel(nn.Module):
    """Frame-synchronous log-mel (hop 160 = one AMBE frame), as neurodec's."""

    def __init__(self, n_fft=512, win=400, n_mels=80):
        super().__init__()
        self.n_fft, self.win = n_fft, win
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels), persistent=False)
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def forward(self, wav):  # [B, N] -> [B, n_mels, N / 160]
        spec = torch.stft(wav, self.n_fft, FRAME, self.win, self.window, center=True, return_complex=True)
        return torch.log(torch.matmul(self.fb, spec.abs() ** 2) + 1e-5)[..., : wav.shape[-1] // FRAME]


class Frontend(nn.Module):
    """bits [B, T, 50] (+-1, 0 for an erased frame, + erasure flag) and the reference log-mel [B, T, 80]
    -> clean-speech log-mel [B, T - lookahead, 80].

    Causal mode: the focal modulation sees past frames only (causal depthwise convolutions; a 512-frame window in
    place of the global average; `window_size` frames), and frame t is read at position t + lookahead, i.e. from the inputs of frames
    <= t + lookahead."""

    def __init__(self, dim=768, focal_window=7, focal_level=3, causal=False, lookahead=0, window_size=512, proj=512):
        super().__init__()
        self.lookahead = lookahead
        self.bit_proj = nn.Sequential(nn.Linear(50, proj), nn.GELU(), nn.Linear(proj, proj))
        self.mel_proj = nn.Sequential(nn.Linear(80, proj), nn.GELU(), nn.Linear(proj, proj))
        self.net = FocalDecoder(input_dim=2 * proj, output_dim=dim, hidden_dims=(dim, dim, dim), upscale_factors=(1, 1, 1),
                                focal_window=focal_window, focal_level=focal_level, focal_factor=2, causal=causal,
                                window_size=window_size, last_window_size=window_size, lookahead_size=0)
        self.head = nn.Linear(dim, 80)
        self.register_buffer("mel_mean", torch.full((80,), -6.0))

    def forward(self, bits, ref_mel):
        h = torch.cat([self.bit_proj(bits), self.mel_proj(ref_mel / 4.0)], -1)
        out, _ = self.net(h)
        T = bits.shape[1] - self.lookahead
        return ref_mel[:, :T] + self.head(out[:, self.lookahead:])


class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        f, v = cfg["frontend"], cfg["vocoder"]
        causal = bool(cfg.get("causal", False))
        self.lookahead = int(f.get("lookahead", 0))
        self.mel = LogMel(n_mels=f["n_mels"])
        self.frontend = Frontend(f["dim"], f["focal_window"], f["focal_level"], causal, self.lookahead,
                                 int(f.get("window_size", 512)), int(f.get("proj", 512)))
        self.proj = conv1d(f["n_mels"], v["cond_dim"], 5, causal)
        self.generator = ISTFTGenerator(v["cond_dim"], samples_per_frame=FRAME, causal=causal, **v["istft"])

    @torch.no_grad()
    def forward(self, bits: torch.Tensor, ref_wav: torch.Tensor) -> torch.Tensor:
        """bits [B, T, 50] (see bit_input), ref_wav [B, 160 T] (the reference decoder's output)
        -> wav [B, 160 (T - lookahead)]."""
        ref_mel = self.mel(ref_wav).transpose(1, 2)
        mel = self.frontend(bits, ref_mel).transpose(1, 2)
        B, _, T = mel.shape
        if self.generator.excitation is not None:  # ignored by the vocoder (neurodec docs/experiments.md)
            f0 = torch.full((B, T), 100.0, device=mel.device)
            return self.generator(self.proj(mel), f0=f0, voiced=torch.zeros(B, T, device=mel.device))
        return self.generator(self.proj(mel))


def bit_input(bits: np.ndarray, lost: np.ndarray) -> np.ndarray:
    """uint8 [T, 49] + bool [T] -> float32 [T, 50]: bits as +-1 (0 for erased frames) and an erasure flag."""
    x = bits.astype(np.float32) * 2 - 1
    x[lost] = 0.0
    return np.concatenate([x, lost[:, None].astype(np.float32)], 1)


def load(model_path: str | Path, device: str = "cpu") -> Decoder:
    """models/<name>.pt (state dict, stored in float16) + models/<name>.json (architecture)."""
    model_path = Path(model_path)
    cfg = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    G = Decoder({**cfg["model"], "causal": cfg.get("causal", False)})
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    G.load_state_dict({k: v.float() for k, v in sd.items()})
    return G.to(device).eval()


def decode(G: Decoder, bits: np.ndarray, lost: np.ndarray | None = None, device: str = "cpu",
           chunk_frames: int = 6000, context: int = 100, seed: int = 0) -> np.ndarray:
    """bits uint8 [T, 49] (ambe_d order) + lost bool [T] -> float32 wav [160 T] in [-1, 1], 8 kHz.
    Same timing as the reference decoder's output. Inputs longer than chunk_frames (2 min) are decoded in chunks
    with `context` frames of overlap on both sides (the reference decoder itself always runs over the whole
    stream). The focal modulation has a global-context level, so chunking changes the output slightly.
    A causal model needs `lookahead` frames after the last one: the end of the stream is padded with erased frames.
    The vocoder's excitation noise is drawn from `seed`."""
    T = len(bits)
    lost = np.zeros(T, bool) if lost is None else np.asarray(lost, bool)
    ref = blip25.decode(bits, lost).astype(np.float32) / 32767
    x = bit_input(bits, lost)
    L = G.lookahead
    if L:
        x = np.concatenate([x, bit_input(np.zeros((L, 49), np.uint8), np.ones(L, bool))])
        ref = np.concatenate([ref, np.zeros(L * FRAME, np.float32)])
    torch.manual_seed(seed)
    out = np.zeros(T * FRAME, np.float32)
    for s in range(0, T, chunk_frames):
        a, e = max(0, s - context), min(T, s + chunk_frames + context) + L
        b = torch.from_numpy(x[a:e])[None].to(device)
        r = torch.from_numpy(ref[a * FRAME: e * FRAME])[None].to(device)
        y = G(b, r)[0].float().cpu().numpy()
        end = min(T, s + chunk_frames)
        out[s * FRAME: end * FRAME] = y[(s - a) * FRAME: (end - a) * FRAME]
    return np.clip(out, -1, 1)
