import torch
from monotonic_align import maximum_path

dist = torch.randn(1, 137, 800).abs()
res = maximum_path(dist, torch.ones_like(dist).bool())
