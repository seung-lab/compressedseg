import os
import os.path
import sys

import click
import compressed_segmentation
import numpy as np

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value
  

@click.group()
def main():
	"""
	Compress and decompress compressed_segmentation .cseg files
	from and to numpy .npy files.

	This implementation can be found at 
	https://github.com/seung-lab/compressedseg.

	This CLI client and Python interface are licensed under the 
	BSD-3 Clause license by William Silversmith at Seung Lab Princeton.
	The encoder is Apache 2.0 licensed by Jeremy Maitin-Shepard
	at Google. The decoder is by Stephen Plaza formerly of Janelia
	Farm Research Center's FlyEM project.
	"""
	pass

@main.command()
@click.option('--block-size', type=Tuple3(), default="8,8,8", help="Compression block size.", show_default=True)
@click.argument("source", nargs=-1)
def compress(source, block_size):
	"""Compress a numpy file to a cseg file."""
	for i in range(len(source)):
		if source[i] == "-":
			source = source[:i] + sys.stdin.readlines() + source[i+1:]
	
	for src in source:
			compress_file(src, block_size)

@main.command()
@click.option('--volume-size', required=True, type=Tuple3(), help="Dimensions of the volume.", show_default=True)
@click.option('--block-size', type=Tuple3(), default="8,8,8", help="Compression step size. No effect on decompression.", show_default=True)
@click.option('--bytes', required=True, type=int, help="Number of bytes per voxel. 4 or 8.", show_default=True)
@click.option('--order', default="C", help="C or Fortran ordering (C or F)", show_default=True)
@click.argument("source", nargs=-1)
def decompress(source, volume_size, block_size, bytes, order):
	"""
	Decompress a cseg file. The dimensions, dtype, and
	block size must be known from information outside
	the file.
	"""
	assert bytes in (4,8)
	assert order in ('C','F')

	dtype = np.uint32 if bytes == 4 else np.uint64

	for i in range(len(source)):
		if source[i] == "-":
			source = source[:i] + sys.stdin.readlines() + source[i+1:]

	for src in source:
		decompress_file(src, volume_size, block_size, dtype, order)

def decompress_file(src, volume_size, block_size, dtype, order):
	with open(src, "rb") as f:
		binary = f.read()

	data = compressed_segmentation.decompress(
		binary, volume_size, dtype, 
		block_size, order
	)
	del binary

	dest = src.replace(".cseg", "")
	_, ext = os.path.splitext(dest)
	
	if ext != ".npy":
		dest += ".npy"

	np.save(dest, data)

	try:
		stat = os.stat(dest)
		if stat.st_size > 0:
			os.remove(src)
		else:
			raise ValueError("File is zero length.")
	except (FileNotFoundError, ValueError) as err:
		print(f"cseg: Unable to write {dest}. Aborting.")
		sys.exit()

def compress_file(src, block_size):
	try:
		data = np.load(src)
	except ValueError:
		print(f"cseg: {src} is not a numpy file.")
		sys.exit()

	binary = compressed_segmentation.compress(data, block_size=block_size)
	del data

	dest = f"{src}.cseg"
	with open(dest, "wb") as f:
		f.write(binary)
	del binary

	try:
		stat = os.stat(dest)
		if stat.st_size > 0:
			os.remove(src)
		else:
			raise ValueError("File is zero length.")
	except (FileNotFoundError, ValueError) as err:
		print(f"cseg: Unable to write {dest}. Aborting.")
		sys.exit()

