"""
Cython binding for the C++ compressed_segmentation
library by Jeremy Maitin-Shepard and Stephen Plaza.

Image label encoding algorithm binding. Compatible with
neuroglancer.

Key methods: compress, decompress

License: BSD 3-Clause

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018 - September 2020
"""

from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from cpython cimport array
import array
import sys
import operator
from functools import reduce

from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

ctypedef fused UINT:
  uint32_t
  uint64_t

cdef extern from "compress_segmentation.h" namespace "compress_segmentation":
  cdef int CompressChannels[Label](
    Label* input, 
    const ptrdiff_t input_strides[4],
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    vector[uint32_t]* output
  )

cdef extern from "decompress_segmentation.h" namespace "compress_segmentation":
  cdef void DecompressChannels[Label](
    const uint32_t* input,
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    const ptrdiff_t strides[4],
    vector[Label]* output
  )

DEFAULT_BLOCK_SIZE = (8,8,8)

def compress(data, block_size=DEFAULT_BLOCK_SIZE, order='C'):
  """
  compress(data, block_size=DEFAULT_BLOCK_SIZE, order='C')

  Compress a uint32 or uint64 3D or 4D numpy array using the
  compressed_segmentation technique.

  data: the numpy array
  block_size: typically (8,8,8). Small enough to be considered
    random access on a GPU, large enough to achieve compression.
  order: 'C' (row-major, 'C', XYZ) or 'F' (column-major, fortran, ZYX)
    memory layout.

  Returns: byte string representing the encoded file
  """
  if len(data.shape) < 4:
    data = data[ :, :, :, np.newaxis ]

  cdef ptrdiff_t volume_size[4] 
  volume_size[:] = data.shape[:4]

  cdef ptrdiff_t block_sizeptr[3]
  block_sizeptr[:] = block_size[:3]

  cdef ptrdiff_t input_strides[3]

  if order == 'F':
    input_strides[:] = [ 
      1,
      volume_size[0],
      volume_size[0] * volume_size[1]
    ]
  else:
    input_strides[:] = [ 
      volume_size[1] * volume_size[2],
      volume_size[2], 
      1
    ]

  cdef uint32_t[:,:,:,:] arr_memview32
  cdef uint64_t[:,:,:,:] arr_memview64

  cdef vector[uint32_t] *output = new vector[uint32_t]()
  cdef int error = 0

  if data.dtype == np.uint32:
    if data.size == 0:
      arr_memview32 = np.zeros((1,1,1,1), dtype=np.uint32)
    else:
      arr_memview32 = data
    error = CompressChannels[uint32_t](
      <uint32_t*>&arr_memview32[0,0,0,0],
      <ptrdiff_t*>input_strides,
      <ptrdiff_t*>volume_size,
      <ptrdiff_t*>block_sizeptr,
      output
    )
  else:
    if data.size == 0:
      arr_memview64 = np.zeros((1,1,1,1), dtype=np.uint64)
    else:
      arr_memview64 = data

    error = CompressChannels[uint64_t](
      <uint64_t*>&arr_memview64[0,0,0,0],
      <ptrdiff_t*>input_strides,
      <ptrdiff_t*>volume_size,
      <ptrdiff_t*>block_sizeptr,
      output
    )

  if error:
    raise OverflowError(
      "The input data were too large and varied and generated a table offset larger than 24-bits.\n"
      "See lookupTableOffset: https://github.com/google/neuroglancer/blob/c9a6b9948dd416997c91e655ec3d67bf6b7e771b/src/neuroglancer/sliceview/compressed_segmentation/README.md#format-specification"
    )

  cdef uint32_t* output_ptr = <uint32_t *>&output[0][0]
  cdef uint32_t[:] vec_view = <uint32_t[:output.size()]>output_ptr

  bytestrout = bytes(vec_view[:])
  del output
  return bytestrout

cdef decompress_helper(
    bytes encoded, volume_size, order, 
    block_size=DEFAULT_BLOCK_SIZE, UINT dummy_dtype = 0
  ):
  
  dtype = np.uint32 if sizeof(UINT) == 4 else np.uint64
  if any([ sz == 0 for sz in volume_size ]):
    return np.zeros(volume_size, dtype=dtype, order=order)
  
  decode_shape = volume_size
  if len(decode_shape) == 3:
    decode_shape = (volume_size[0], volume_size[1], volume_size[2], 1)

  cdef unsigned char *encodedptr = <unsigned char*>encoded
  cdef uint32_t* uintencodedptr = <uint32_t*>encodedptr;
  cdef ptrdiff_t[4] volsize = decode_shape
  cdef ptrdiff_t[3] blksize = block_size
  cdef ptrdiff_t[4] strides = [ 
    1, 
    volsize[0], 
    volsize[0] * volsize[1], 
    volsize[0] * volsize[1] * volsize[2] 
  ]

  if order == 'C':
    strides[0] = volsize[1] * volsize[2] * volsize[3]
    strides[1] = volsize[2] * volsize[3]
    strides[2] = volsize[3]
    strides[3] = 1

  cdef vector[UINT] *output = new vector[UINT]()

  DecompressChannels(
    uintencodedptr,
    volsize,
    blksize,
    strides,
    output
  )
  
  cdef UINT* output_ptr = <UINT*>&output[0][0]
  cdef UINT[:] vec_view = <UINT[:output.size()]>output_ptr

  # possible double free issue
  # The buffer gets loaded into numpy, but not the vector<uint64_t>
  # So when numpy clears the buffer, the vector object remains
  # Maybe we should make a copy of the vector into a regular array.

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(vec_view[:])
  del output
  return np.frombuffer(buf, dtype=dtype).reshape( volume_size, order=order )

def decompress(
    bytes encoded, volume_size, dtype, 
    block_size=DEFAULT_BLOCK_SIZE, order='C'
  ):
  """
  decompress(
    bytes encoded, volume_size, dtype, 
    block_size=DEFAULT_BLOCK_SIZE, order='C'
  )

  Decode a compressed_segmentation file into a numpy array.

  encoded: the file as a byte string
  volume_size: tuple with x,y,z dimensions
  dtype: np.uint32 or np.uint64
  block_size: typically (8,8,8), the block size the file was encoded with.
  order: 'C' (XYZ) or 'F' (ZYX)

  Returns: 3D or 4D numpy array
  """
  dtype = np.dtype(dtype)
  if dtype == np.uint32:
    return decompress_helper(encoded, volume_size, order, block_size, <uint32_t>0)
  elif dtype == np.uint64:
    return decompress_helper(encoded, volume_size, order, block_size, <uint64_t>0)
  else:
    raise TypeError("dtype ({}) must be one of uint32 or uint64.".format(dtype))

def labels(
  bytes encoded, volume_size, dtype, 
  block_size=DEFAULT_BLOCK_SIZE
):
  """Extract labels without decompressing."""
  volume_size = np.array(volume_size)
  block_size = np.array(block_size)

  grid_size = np.ceil(volume_size / block_size).astype(np.uint64)
  cdef size_t num_headers = reduce(operator.mul, grid_size)
  cdef size_t header_bytes = 8 * num_headers

  encoded = encoded[4:] # skip the channel length
  cdef np.ndarray[uint64_t] headers = np.frombuffer(encoded[:header_bytes], dtype=np.uint64)
  cdef np.ndarray[uint32_t] data = np.frombuffer(encoded, dtype=np.uint32)

  cdef np.ndarray[uint32_t] offsets = np.zeros((2*num_headers,), dtype=np.uint32)

  cdef size_t i = 0
  cdef size_t lookup_table_offset = 0
  cdef size_t encoded_values_offset = 0
  for i in range(num_headers):
    lookup_table_offset = headers[i] & 0xffffff
    encoded_values_offset = headers[i] >> 32
    offsets[2 * i] = lookup_table_offset
    offsets[2 * i + 1] = encoded_values_offset

  # use unique rather than simply sort b/c
  # label offsets can be reused.
  offsets = np.unique(offsets)

  labels = np.zeros((0,), dtype=dtype)

  cdef size_t dtype_bytes = np.dtype(dtype).itemsize
  cdef size_t start = 0
  cdef size_t end = 0

  cdef int64_t idx = 0
  cdef int64_t size = offsets.size - 1

  cdef np.ndarray[uint32_t, ndim=2] index = np.zeros((num_headers, 2), dtype=np.uint32)

  for i in range(num_headers):
    lookup_table_offset = headers[i] & 0xffffff
    idx = _search(offsets, lookup_table_offset)
    if idx == -1:
      raise IndexError(f"Unable to locate value: {lookup_table_offset}")
    elif idx == size:
      index[i, 0] = offsets[idx]
      index[i, 1] = data.size
    else:
      index[i, 0] = offsets[idx]
      index[i, 1] = offsets[idx+1]

  labels = np.concatenate([ 
    data[index[idx,0]:index[idx,1]]
    for idx in range(num_headers) 
  ]).view(dtype)

  return np.unique(labels)

cdef int64_t _search(np.ndarray[uint32_t] offsets, uint32_t value):
  cdef size_t first = 0
  cdef size_t last = offsets.size - 1
  cdef size_t middle = (first // 2 + last // 2)

  while (last - first) > 1:
    if offsets[middle] == value:
      return middle
    elif offsets[middle] > value:
      last = middle
    else:
      first = middle

    middle = (first // 2 + last // 2 + ((first & 0b1) + (last & 0b1)) // 2)

  if offsets[first] == value:
    return first

  if offsets[last] == value:
    return last

  return -1



