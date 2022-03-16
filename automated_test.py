import pytest
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

def check_compressed_headers(compressed, shape, block_size):
    compressed = np.frombuffer(compressed, dtype=np.uint32)
    # test headers for correctness
    assert compressed[0] == 1 # only 1 channel
    # 64 bit block header 
    # encoded bits (8 bit), lookup table offset (24 bit), encodedValuesOffset (32)
    grid = np.ceil(
        np.array(shape, dtype=np.float32) / np.array(block_size, dtype=np.float32)
    ).astype(np.uint32)
    for i in range(np.prod(grid)):
      encodedbits = (compressed[2*i + 1] & 0xff000000) >> 24
      table_offset = compressed[2*i + 1] & 0x00ffffff
      encoded_offset = compressed[(2*i + 1) + 1]

      assert encodedbits in (0, 1, 2, 4, 8, 16, 32)
      assert table_offset < len(compressed)
      assert encoded_offset < len(compressed)

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('variation', (1,2,4,8,16,32,64,128,256,512,1024))
@pytest.mark.parametrize('block_size', [ (2,2,2), (4,4,4), (8,8,8) ])
def test_recover_random(dtype, order, block_size, variation):
  for _ in range(3):
    sx = random.randint(0, 256)
    sy = random.randint(0, 256)
    sz = random.randint(0, 128)

    labels = np.random.randint(variation, size=(sx, sy, sz), dtype=dtype)
    labels = np.copy(labels, order=order)
    compressed = cseg.compress(labels, block_size=block_size, order=order)

    check_compressed_headers(compressed, [sx,sy,sz], block_size)

    recovered = cseg.decompress(
        compressed, (sx, sy, sz), 
        block_size=block_size, dtype=dtype, 
        order=order
    )
    N = np.sum(recovered != labels)
    if N > 0:
      print("Non-matching: ", N)

    assert np.all(labels == recovered)

def test_empty_labels():
    labels = np.zeros((0,), dtype=np.uint32).reshape((0,0,0))
    compressed = cseg.compress(labels, block_size=(8,8,8))

    try:
        cseg.labels(b'', block_size=(8,8,8), shape=(0,0,0), dtype=np.uint32)
        assert False
    except cseg.DecodeError:
        pass
    
    clabels = cseg.labels(compressed, block_size=(8,8,8), shape=(0,0,0), dtype=np.uint32)
    assert all(clabels == labels)

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('variation', (1,2,4,8,16,32,64,128,256,512,1024))
@pytest.mark.parametrize('block_size', [ (2,2,2), (4,4,4), (8,8,8) ])
def test_labels(dtype, order, variation, block_size):

  for _ in range(3):
    sx = random.randint(0, 256)
    sy = random.randint(0, 256)
    sz = random.randint(0, 128)

    labels = np.random.randint(variation, size=(sx, sy, sz), dtype=dtype)
    labels = np.copy(labels, order=order)
    compressed = cseg.compress(labels, block_size=block_size, order=order)

    uniq = np.unique(labels)
    uniql = cseg.labels(
        compressed, 
        shape=(sx,sy,sz), 
        dtype=dtype,
        block_size=block_size,
    )
    assert np.all(uniq == uniql)

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
def test_table_offset_error_random(dtype, order):
    sx = 300
    sy = 300
    sz = 300

    labels = np.random.randint(10000, size=(sx, sy, sz), dtype=dtype)

    try:
        cseg.compress(labels, order=order)
        assert False, "An OverflowError should have been triggered."
    except OverflowError:
        pass

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('order', ("C", "F"))
def test_table_offset_error_sequence(dtype, order):
    sx = 256
    sy = 256
    sz = 32

    labels = np.arange(0, sx*sy*sz, dtype=dtype).reshape((sx,sy,sz), order=order)
    compressed = cseg.compress(labels, order=order)
    recovered = cseg.decompress(compressed, (sx, sy, sz), dtype=dtype, order=order)

    assert np.all(labels == recovered)

@pytest.mark.parametrize('dtype', (np.uint32, np.uint64))
@pytest.mark.parametrize('block_size', [ (2,2,2), (4,4,4), (8,8,8) ])
def test_random_access(dtype, block_size):
    sx = 100
    sy = 100
    sz = 100

    labels = np.random.randint(10000, size=(sx, sy, sz), dtype=dtype)
    binary = cseg.compress(labels, block_size=block_size)
    arr = cseg.CompressedSegmentationArray(
        binary, shape=(sx,sy,sz), dtype=dtype, block_size=block_size
    )

    for i in range(10):
        x,y,z = np.random.randint(0, sx, size=(3,), dtype=int)
        assert arr[x,y,z] == labels[x,y,z]

    labels = np.zeros((sx, sy, sz), dtype=dtype)
    binary = cseg.compress(labels, block_size=block_size)
    arr = cseg.CompressedSegmentationArray(
        binary, shape=(sx,sy,sz), dtype=dtype, block_size=block_size
    )

    for i in range(10):
        x,y,z = np.random.randint(0, sx, size=(3,), dtype=int)
        assert arr[x,y,z] == labels[x,y,z]

@pytest.mark.parametrize('dtype', [ np.uint32, np.uint64 ])
@pytest.mark.parametrize('preserve_missing_labels', [ True, False ])
def test_remap(dtype, preserve_missing_labels):
    shape = (61,63,67)
    labels = np.random.randint(0, 15, size=shape).astype(dtype)

    remap = { i: i+20 for i in range(15) }
    binary = cseg.compress(labels)
    recovered = cseg.decompress(binary, shape, dtype)

    assert np.all(labels == recovered)
    assert np.all(cseg.labels(binary, shape, dtype) == list(range(15)))

    binary2 = cseg.remap(
        binary, shape, dtype, 
        mapping=remap, 
        preserve_missing_labels=preserve_missing_labels,
    )
    assert np.all(cseg.labels(binary2, shape, dtype) == list(range(20, 35)))


