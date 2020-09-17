import pytest
from cloudvolume import view
import compressed_segmentation as cseg 
import numpy as np
import random

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
def test_zero_size(dtype, order):
    sx = 0
    sy = random.randint(1, 512)
    sz = random.randint(1, 512)    

    labels = np.random.randint(10, size=(sx, sy, sz), dtype=dtype)
    compressed = cseg.compress(labels, order=order)
    recovered = cseg.decompress(compressed, (sx, sy, sz), dtype=dtype, order=order)

    assert labels.shape == recovered.shape

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

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
def test_table_offset_error(dtype, order):
    sx = 300
    sy = 300
    sz = 300

    labels = np.random.randint(10000, size=(sx, sy, sz), dtype=dtype)

    try:
        cseg.compress(labels, order=order)
        assert False, "An OverflowError should have been triggered."
    except OverflowError:
        pass
