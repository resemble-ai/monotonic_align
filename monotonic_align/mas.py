import numpy as np
import torch
from monotonic_align.core import maximum_path_c
from monotonic_align.core2 import maximum_path_c2
from monotonic_align.core2eps import maximum_path_c2eps
from monotonic_align.core_my import maximum_path_cmy


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


def maximum_path(value, mask=None, topology="1-step"):
  """
  Cython optimised version of Monotonic Alignment Search
  Returns the most likely alignment for the given log-likelihood matrix.

  Args:
    value: the log-likelihood matrix.
      Its (i, j)-th entry contains the log-likelihood of the j-th latent variable
      for the given i-th prior mean and variance.
      (dtype=float, shape=[batch_size, text_length, latent_variable_length])
    mask: same shape as `value`
    topology: "1-step", "2-step", "1-epsilon"
    .. math::
    value_{i,j} = log N(f(z)_{j}; \mu_{i}, \sigma_{i})

  Returns:
    path: the most likely alignment.
    (dtype=float, shape=[text_length, latent_variable_length])
  """
  if mask is None:
    mask = torch.ones_like(value)

  device = value.device
  dtype = value.dtype

  value = value * mask
  value = value.data.cpu().numpy().astype(np.float32)
  mask = mask.data.cpu().numpy()

  path = np.zeros_like(value).astype(np.int32)
  t_x_max = mask.sum(1)[:, 0].astype(np.int32)
  t_y_max = mask.sum(2)[:, 0].astype(np.int32)

  if topology == "1-step":
    maximum_path_c(path, value, t_x_max, t_y_max)

  elif topology == "2-step":
    maximum_path_c2(path, value, t_x_max, t_y_max)

  elif topology == "1-epsilon":
    maximum_path_c2eps(path, value, t_x_max, t_y_max)

  elif topology == "1-step-my":
    maximum_path_cmy(path, value, t_x_max, t_y_max)

  else:
    raise ValueError(f"Unknown topology: {topology}")

  return torch.from_numpy(path).to(device=device, dtype=dtype), value
