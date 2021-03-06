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
from libc.stdint cimport uint32_t, uint64_t
from cpython cimport array
import array
import sys

from libcpp.vector cimport vector

cimport numpy as cnp
import numpy as np

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

cdef decompress_helper32(
    bytes encoded, volume_size, dtype, order, 
    block_size=DEFAULT_BLOCK_SIZE
  ):

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

  cdef vector[uint32_t] *output = new vector[uint32_t]()

  DecompressChannels[uint32_t](
    uintencodedptr,
    volsize,
    blksize,
    strides,
    output
  )
  
  cdef uint32_t* output_ptr = <uint32_t*>&output[0][0]
  cdef uint32_t[:] vec_view = <uint32_t[:output.size()]>output_ptr

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(vec_view[:])
  del output
  return np.frombuffer(buf, dtype=dtype).reshape( volume_size, order=order )

cdef decompress_helper64(
    bytes encoded, volume_size, dtype, order, 
    block_size=DEFAULT_BLOCK_SIZE
  ):
  
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

  cdef vector[uint64_t] *output = new vector[uint64_t]()

  DecompressChannels[uint64_t](
    uintencodedptr,
    volsize,
    blksize,
    strides,
    output
  )
  
  cdef uint64_t* output_ptr = <uint64_t*>&output[0][0]
  cdef uint64_t[:] vec_view = <uint64_t[:output.size()]>output_ptr

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
    return decompress_helper32(encoded, volume_size, dtype, order, block_size)
  elif dtype == np.uint64:
    return decompress_helper64(encoded, volume_size, dtype, order, block_size)
  else:
    raise TypeError("dtype ({}) must be one of uint32 or uint64.".format(dtype))



