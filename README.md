Adapted from the MAS in [Glow-TTS](https://github.com/jaywalnut310/glow-tts/tree/master/monotonic_align). I made it installable and added variants.

# Installation
```
pip install git+https://github.com/resemble-ai/monotonic_align.git
```
Installing `monotonic_align` doesn't require torch, but using ``monotonic_align`` will.
Please install PyTorch yourself, as its installation differ from system to system.


# How to Use
```python
# Suppose you have:
# 1. a probability matrix of size (batch_size=B, symbol_len=S, mel_lens=T)
#    NOTE: a similarity matrix (a higher score means better) or negative cost will do
#          but may have issues.
# 2. an array of symbol lengths `symbol_lens` of size (batch_size=B)
# 3. an array of mel-spectrogram lengths `mel_lens` of size (batch_size=B)

from monotonic_align import mask_from_lens, maximum_path
mask_ST = mask_from_lens(similarity, symbol_lens, mel_lens)
alignment = maximum_path(similarity, mask_ST)  # (B, S, T)

# NOTE:
# - If `mask` is not specified, the default mask is `True` for all elements.
# - You can specify `topology` if you want to use other variants of alignment algorithms.
```
