Adapted from the MAS in [Glow-TTS](https://github.com/jaywalnut310/glow-tts/tree/master/monotonic_align). I made it installable and added variants.

# Installation
```bash
git clone https://github.com/resemble-ai/monotonic_align.git
cd monotonic_align
pip install .
```
Installing `monotonic_align` doesn't require torch, but using ``monotonic_align`` will.
Please install PyTorch yourself, as its installation differ from system to system.


# How to Use
```python
# This module is used to find the path that "maximizes" the score along the path.
# If your input is "cost" or "distance", please make sure you put a minus sign to it.

# Suppose similarity.shape is (batch_size=1, symbol_len=S, mel_lens=T)
from monotonic_align import maximum_path
alignment = maximum_path(similarity)  # (1, S, T)


# Make sure to specify the `mask` argument in batch mode.
alignment = maximum_path(similarity, mask)  # (B, S, T)


# You can use the utility function `mask_from_len` for the mask:
from monotonic_align import mask_from_lens
mask_ST = mask_from_lens(similarity, symbol_lens, mel_lens)
alignment = maximum_path(similarity, mask_ST)  # (B, S, T)
```
