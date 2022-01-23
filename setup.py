import os
import setuptools
import sys

# NOTE: Run if _compressed_segmentation.cpp does not exist:
# cython -3 --fast-fail -v --cplus -I./include src/compressed_segmentation.pyx

import numpy as np

join = os.path.join

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3'
  ]

setuptools.setup(
  setup_requires=['numpy', 'pbr'],
  ext_modules=[
    setuptools.Extension(
        'compressed_segmentation',
        optional=True,
        sources=[ join('src', x) for x in ( 
            'compress_segmentation.cc', 'decompress_segmentation.cc',
            'compressed_segmentation.cpp'
        )],
        language='c++',
        include_dirs=[ 'include', np.get_include() ],
        extra_compile_args=extra_compile_args,
    ),
  ],
  long_description_content_type="text/markdown",
  pbr=True,
  entry_points={
    "console_scripts": [
      "cseg=cseg_cli:main"
    ],
  },
)


