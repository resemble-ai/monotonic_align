from typing import overload
import numpy as np
import torch
from monotonic_align.core import maximum_path_c


def mask_from_len(lens: torch.Tensor, max_len=None):
    """
    Make a `mask` from lens.
    
    :param inputs: (B, T, D)
    :param lens: (B)
    
    :return:
        `mask`: (B, T)
    """
    if max_len is None:
      max_len = lens.max()
    index = torch.arange(max_len).to(lens).view(1, -1)
    return index < lens.unsqueeze(1)  # (B, T)


def mask_from_lens(
  similarity: torch.Tensor,
  symbol_lens: torch.Tensor,
  mel_lens: torch.Tensor,
):
  """
  :param similarity: (B, S, T)
  :param symbol_lens: (B,)
  :param mel_lens: (B,)
  """
  _, S, T = similarity.size()
  mask_S = mask_from_len(symbol_lens, S)
  mask_T = mask_from_len(mel_lens, T)
  mask_ST = mask_S.unsqueeze(2) * mask_T.unsqueeze(1)
  return mask_ST.to(similarity)


def maximum_path(value, mask=None):  
  """ Cython optimised version.
  value: [b, t_x, t_y]
  mask: [b, t_x, t_y]
  """
  if mask is None:
    mask = torch.zeros_like(value)

  value = value * mask
  device = value.device
  dtype = value.dtype
  value = value.data.cpu().numpy().astype(np.float32)
  path = np.zeros_like(value).astype(np.int32)
  mask = mask.data.cpu().numpy()
  t_x_max = mask.sum(1)[:, 0].astype(np.int32)
  t_y_max = mask.sum(2)[:, 0].astype(np.int32)
  maximum_path_c(path, value, t_x_max, t_y_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)
