import pytest

import compressed_segmentation as cseg 
import numpy as np
import random
from tqdm import tqdm

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
def test_recover_random(dtype, order):
  for _ in range(100):
    sx = random.randint(0, 100)
    sy = random.randint(0, 100)
    sz = random.randint(0, 100)

    labels = np.random.randint(255, size=(sx, sy, sz), dtype=dtype)
    compressed = cseg.compress(labels, order=order)
    recovered = cseg.decompress(compressed, (sx, sy, sz), dtype=dtype, order=order)

    assert type(compressed) in (bytes, str)
    assert np.all(recovered == labels)

