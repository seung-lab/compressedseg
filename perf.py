import pytest
import compressed_segmentation as cseg 
import numpy as np
import random
import time


def measure(dtype, order, N=20):
  csec = 0
  dsec = 0
  voxels = 0
  for i in range(N):
    sx = random.randint(0, 256)
    sy = random.randint(0, 256)
    sz = random.randint(0, 128)
    
    voxels += sx * sy * sz

    labels = np.random.randint(10000, size=(sx, sy, sz), dtype=dtype)
    labels = np.arange(0, sx*sy*sz, dtype=dtype).reshape((sx,sy,sz), order=order)
    
    s = time.time()
    compressed = cseg.compress(labels, order=order)
    csec += time.time() - s

    s = time.time()
    recovered = cseg.decompress(compressed, (sx, sy, sz), dtype=dtype, order=order)
    dsec += time.time() - s

    assert np.all(labels == recovered)
    assert labels.flags.f_contiguous == recovered.flags.f_contiguous
    assert labels.flags.c_contiguous == recovered.flags.c_contiguous

  print("MVx: {:.2f}, {}, order: {}".format(voxels / 1e6, dtype, order))
  print("Compression: {:.2f} sec :: {:.2f} MVx/sec".format(csec, voxels / csec / 1e6))
  print("Decompression: {:.2f} sec :: {:.2f} MVx/sec".format(dsec, voxels / dsec / 1e6))


measure(np.uint32, 'C')
measure(np.uint32, 'F')
measure(np.uint64, 'C')
measure(np.uint64, 'F')