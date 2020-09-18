/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
 

#include "decompress_segmentation.h"

#include <algorithm>
#include <unordered_map>
#include <iostream>

using std::min;

namespace compress_segmentation {

constexpr size_t dec_kBlockHeaderSize = 2;

template <class Label>
void DecompressChannel(
	const uint32_t* input,
	const ptrdiff_t volume_size[3],
	const ptrdiff_t block_size[3],
	const ptrdiff_t strides[4],
	std::vector<Label>* output,
	const ptrdiff_t channel
) {
	const size_t table_entry_size = (sizeof(Label) + sizeof(uint32_t) - 1) / sizeof(uint32_t);

	// determine number of grids for volume specified and block size
	// (must match what was encoded) 
	ptrdiff_t grid_size[3];
	for (size_t i = 0; i < 3; ++i) {
		grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
	}

	ptrdiff_t block[3];
	for (block[2] = 0; block[2] < grid_size[2]; ++block[2]) {
		for (block[1] = 0; block[1] < grid_size[1]; ++block[1]) {
			for (block[0] = 0; block[0] < grid_size[0]; ++block[0]) {
				const size_t block_offset =
					block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);
			
				size_t encoded_bits, tableoffset, encoded_value_start;
				tableoffset = input[block_offset * dec_kBlockHeaderSize] & 0xffffff;
				encoded_bits = (input[block_offset * dec_kBlockHeaderSize] >> 24) & 0xff;
				encoded_value_start = input[block_offset * dec_kBlockHeaderSize + 1];

				// find absolute positions in output array (+ base_offset)
				size_t xmin = block[0]*block_size[0];
				size_t xmax = min(xmin + block_size[0], size_t(volume_size[0]));

				size_t ymin = block[1]*block_size[1];
				size_t ymax = min(ymin + block_size[1], size_t(volume_size[1]));

				size_t zmin = block[2]*block_size[2];
				size_t zmax = min(zmin + block_size[2], size_t(volume_size[2]));

				uint64_t bitmask = (1 << encoded_bits) - 1;
				for (size_t z = zmin; z < zmax; ++z) {
					for (size_t y = ymin; y < ymax; ++y) {
						size_t base_outindex = strides[1] * y + strides[2] * z + strides[3] * channel;
						size_t bitpos = (
							block_size[0] * ((z-zmin) * (block_size[1]) 
							+ (y-ymin)) * encoded_bits
						);

						for (size_t x = xmin; x < xmax; ++x) {
							size_t outindex = base_outindex + strides[0] * x;
							size_t bitshift = bitpos % 32;
							size_t arraypos = bitpos / (32);
							size_t bitval = 0;
							if (encoded_bits > 0) {
								bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
							}
							Label val = input[tableoffset + bitval*table_entry_size];
							if (table_entry_size == 2) {
								val |= uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
							}
							(*output)[outindex] = val;
							bitpos += encoded_bits; 
						}
					}
				}
		  }
		}
  }
}

template <class Label>
void DecompressChannels(
	const uint32_t* input,
	const ptrdiff_t volume_size[4],
	const ptrdiff_t block_size[3],
	const ptrdiff_t strides[4],
	std::vector<Label>* output
) {

  /*
  A simple encoding is used to store multiple channels of compressed segmentation data 
  (assumed to have the same x, y, and z dimensions and compression block size) together. 
  The number of channels, num_channels, is assumed to be known.

  The header consists of num_channels little-endian 32-bit unsigned integers specifying 
  the offset, in 32-bit units from the start of the file, at which the data for each 
  channel begins. The channels should be packed in order, and without any padding. 
  The offset specified in the header for the first channel must be equal to num_channels.

  In the special case that this format is used to encode just a single compressed 
  segmentation channel, the compressed segmentation data is simply prefixed with a 
  single 1 value (encoded as a little-endian 32-bit unsigned integer).
  */

  size_t voxels = volume_size[0] * volume_size[1] * volume_size[2];
  output->resize(voxels * volume_size[3]);

  for (size_t channel_i = 0; channel_i < volume_size[3]; ++channel_i) {
		DecompressChannel(
			input + input[channel_i], volume_size,
			block_size, strides, output, channel_i
		);
  }
}

#define DO_INSTANTIATE(Label)                                        \
  template void DecompressChannel<Label>(                              \
	  const uint32_t* input, const ptrdiff_t volume_size[3],       \
	  const ptrdiff_t block_size[3], \
	  const ptrdiff_t strides[4], \
	  std::vector<Label>* output, \
	  const ptrdiff_t channel);                                \
  template void DecompressChannels<Label>(                             \
	  const uint32_t* input, const ptrdiff_t volume_size[4],            \
	  const ptrdiff_t block_size[3], \
	  const ptrdiff_t strides[4], \
	  std::vector<Label>* output);                                \
/**/

DO_INSTANTIATE(uint32_t)
DO_INSTANTIATE(uint64_t)

#undef DO_INSTANTIATE

}  // namespace compress_segmentation
