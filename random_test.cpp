#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "src/compress_segmentation.cc"
#include "src/decompress_segmentation.cc"

void trial() {
	int sx, sy, sz;

	sx = rand() % 512;
	sy = rand() % 512;
	sz = rand() % 512;

	printf("<%d,%d,%d>\n", sx,sy,sz);

	size_t voxels = sx * sy * sz;

	uint32_t *labels = new uint32_t[voxels]();
	for (size_t i = 0; i < voxels; i++) {
		labels[i] = rand() % 10;
	}

	const ptrdiff_t strides[3] = { 1, sx, sx * sy };
	const ptrdiff_t volume_size[3] = { sx, sy, sz };
	const ptrdiff_t block_size[3] = {8,8,8};
	std::vector<uint32_t> *output = new std::vector<uint32_t>();

	compress_segmentation::CompressChannel<uint32_t>(labels, strides, volume_size, block_size, output);
	// input, volume, block_size, output

	std::vector<uint32_t> *recovered = new std::vector<uint32_t>();
	auto* d_input = output->data();
	compress_segmentation::DecompressChannel<uint32_t>(d_input, volume_size, block_size, recovered);

	delete[] labels;
	delete output;
	delete recovered;
}


int main() {
	for (int n = 0; n < 100; n++) {
		if (n % 500 == 0) {
			printf("n: %d\n", n);
		}
		trial();
	}

	return 0;
}