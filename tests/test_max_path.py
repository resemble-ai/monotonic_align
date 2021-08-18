import torch
from monotonic_align import maximum_path, mask_from_lens


# Example 1
similarity = torch.randn(1, 137, 800).abs()
alignments = maximum_path(similarity)


# Example 2
B = 16   # batch_size
S = 45   # max_symbol_len
T = 500  # max_mel_len
M = 80   # num_mels

symbol_embs = torch.randn(B, S, M)
symbol_lens = torch.randint(1, S, size=[B])

mels = torch.randn(B, T, M)
mel_lens = torch.randint(1, T, size=[B])

similarity = - (symbol_embs.unsqueeze(2) - mels.unsqueeze(1)).pow(2).sum(-1)
mask_ST = mask_from_lens(similarity, symbol_lens, mel_lens)
alignments = maximum_path(similarity, mask_ST)
