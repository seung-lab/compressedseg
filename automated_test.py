import pytest
from cloudvolume import view
import compressed_segmentation as cseg 
import numpy as np
import random

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('variation', (1,2,4,8,16,32,64,128,256,512,1024))
def test_recover_random(dtype, order, variation):
  for _ in range(3):
    sx = random.randint(0, 512)
    sy = random.randint(0, 512)
    sz = random.randint(0, 512)

    labels = np.random.randint(variation, size=(sx, sy, sz), dtype=dtype)
    compressed = cseg.compress(labels, order=order)
    recovered = cseg.decompress(compressed, (sx, sy, sz), dtype=dtype, order=order)

    N = np.sum(recovered != labels)
    if N > 0:
      print("Non-matching: ", N)

    assert np.all(labels == recovered)