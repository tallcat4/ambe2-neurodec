"""AMBE+2 half-rate (2450 bps) neural decoder: 49-bit voice frames -> speech."""
from .model import Decoder, decode, load

__all__ = ["Decoder", "decode", "load"]
