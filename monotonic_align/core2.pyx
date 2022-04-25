import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each2(int[:,::1] path, float[:,::1] value, int n_symbols, int n_frames, float max_neg_val) nogil:
  cdef int x
  cdef int y

  cdef float v_prev_symbol
  cdef float v_prev_state
  cdef float v_cur
  cdef float tmp
  cdef int index = n_symbols - 1

  cdef int frame_index
  cdef int symbol_index
  cdef float stay = 0.0
  cdef float eps_transition = 0.0
  cdef float symbol_transition = 0.0
  cdef float max_ = 0.0
  cdef float max_val = 0.0

  # dynamic programming
  for frame_index in range(n_frames):
    for symbol_index in range(max(0, n_symbols + frame_index - n_frames), min(n_symbols, frame_index + 1)):
      # w/o an auxiliary matrix (one for accumulation and one for raw score value)
      # the number of visited symbols must be < frame index
      if symbol_index == frame_index:
        v_cur = max_neg_val
      else:
        v_cur = value[symbol_index, frame_index - 1]


      # Extra: v_prev_symbol
      if symbol_index >= 2 and symbol_index % 2 == 0:
        # if symbol_index % 2 == 0:  # regular/initial state of a phone
        v_prev_symbol = value[symbol_index - 2, frame_index - 1]
      else:  # epsilon state
          # v_prev_symbol = value[symbol_index - 2, frame_index - 1]
        # else:
        v_prev_symbol = max_neg_val


      # Corner cases at the start
      if symbol_index == 0:
        # score of the 1st row is 0, meaning that
        # a valid path can start from any frame at the 1st symbol.
        if frame_index == 0:
          v_prev_state = 0.
        # score of all other symbols cannot be the start. (score = -inf)
        else:
          v_prev_state = max_neg_val
      # ordination cases
      else:
        v_prev_state = value[symbol_index-1, frame_index-1]

      # value after transition
      max_ = max(v_cur, v_prev_state)
      value[symbol_index, frame_index] = max(max_, v_prev_symbol) + value[symbol_index, frame_index]

  # backtracking (indicator matrix)
  for frame_index in range(n_frames - 1, -1, 1):
    path[index, frame_index] = 1
    if index != 0:
      # must transition due to monotonic constraint
      if index == frame_index:
        index = index - 1

      # transition priority: phone > epi > stay
      stay = value[index, frame_index - 1]
      eps_transition = value[index - 1, frame_index - 1]
      symbol_transition = value[index - 2, frame_index - 1]

      max_ = max(stay, eps_transition)
      max_val = max(max_, symbol_transition)
      if symbol_transition == max_val:
        index = index - 2
      else:
        if eps_transition == max_val:
          index = index - 1
        else:
          if stay == max_val:
            index = index - 1
          else:
            index = index - 0

  # backtracking (for the rest of the 2xT matrix)
  # for y in range(t_y - 1, -1, -1):
  for y in range(1, -1, -1):
    path[index, y] = 1
    if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
      index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c2(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_xs, int[::1] t_ys, float max_neg_val=-1e9) nogil:
  cdef int b = values.shape[0]

  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each2(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)