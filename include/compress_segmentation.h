/**
 * @license
 * Copyright 2016 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified by: Stephen Plaza 2017 under LICENSE_JANELIA.txt
 */

// Implements encoding into the compressed segmentation format described at
// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation.
//
// Only uint32 and uint64 volumes are supported.

// Compress a 3-D label array by splitting in a grid of fixed-size blocks, and
// encoding each block using a per-block table of label values.  The number of
// bits used to encode the value within each block depends on the size of the
// table, i.e. the number of distinct uint64 values within that block.  The
// number of BITS is required to be either 0, or a power of 2, i.e. 0, 1, 2, 4,
// 8, 16.
//
// The format consists of a block index containing a block header for each
// block, followed by the encoded block values, followed by the table that maps
// encoded indices to uint32 or uint64 label values.  Blocks are numbered as:
//   x + grid_size.x() * (y + grid_size.y() * z).
//
// Overall file format:
//
//   [block header] * <number of blocks>
//   [encoded values]
//   [value table]
//
// The format of each block header is:
//
//   table_base_offset : 24-bit LE integer
//   encoding_bits : 8-bit unsigned integer
//
//   encoded_value_base_offset : 24-bit LE integer
//   padding : 8 bits
//
//
// The encoded_value_base_offset specifies the offset in 32-bit units from the
// start of the file to the first 32-bit unit containing encoded values for the
// block.
//
// The table_base_offset specifies the offset in 32-bit units from the start of
// the file to the first table entry for the block.
//
// If multiple blocks have exactly the same set of encoded values, the same
// value table will be shared by both blocks.

#ifndef COMPRESS_SEGMENTATION_H_
#define COMPRESS_SEGMENTATION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace compress_segmentation {


// Encodes a single channel.
//
// Args:
//   input: Pointer to the first element
//
//   input_strides: Stride in uint64 units between consecutive elements in the
//       x, y, and z dimensions.
//
//   volume_size: Extent of the x, y, and z dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector to which output will be appended.
template <class Label>
int CompressChannel(const Label* input, const ptrdiff_t input_strides[3],
                     const ptrdiff_t volume_size[3],
                     const ptrdiff_t block_size[3],
                     std::vector<uint32_t>* output);

// Encodes multiple channels.
//
// Each channel is encoded independently.
//
// The output starts with num_channels (=volume_size[3]) uint32 values
// specifying the starting offset of the encoding of each channel (the first
// offset will always equal num_channels).
//
// Args:
//
//   input: Pointer to the first element.
//
//   input_strides: Stride in uint64 units between consecutive elements in the
//       x, y, z, and channel dimensions.
//
//   volume_size: Extent of the x, y, z, and channel dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector where output will be stored.  Any existing content is
//       cleared.
template <class Label>
int CompressChannels(const Label* input, const ptrdiff_t input_strides[4],
                      const ptrdiff_t volume_size[4],
                      const ptrdiff_t block_size[3],
                      std::vector<uint32_t>* output);

}  // namespace compress_segmentation

#endif  // COMPRESS_SEGMENTATION_H_
