
# Install the package for test
# option 1: python setup.py install
# option 2: pip install submodules/cpu-adam

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
os.path.dirname(os.path.abspath(__file__))

sources = [
    "csrc/cpu_adam.cpp",
    "csrc/cpu_adam_impl.cpp",
    # "csrc/cpu_adam.h",
    # "csrc/simd.h",
]

setup(name='cpu_adam',
      packages=['cpu_adam'],
      ext_modules=[
        cpp_extension.CppExtension(
          # name='cpu_adam',
          name='cpu_adam._C',
          sources=sources,
          extra_compile_args=[
            '-D__AVX256__', 
            '-O3', 
            '-std=c++17', 
            '-g',
            '-Wno-reorder',
            '-march=native',
            '-fopenmp',
            '-D__DISABLE_CUDA__',
            # This code is not mature enough. 
            # To learn how to set compilation flags here, refer to:
            # https://github.com/microsoft/DeepSpeed/blob/45b363504e716058647607a79e81e98786098cf5/op_builder/builder.py#L773
          ],
          # headers=[
          #     "csrc/simd.h",
          #     "csrc/cpu_adam.h",
          # ],
         )
        ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)


# This is an example
# NOTE: before `import lltm_cpp`, we should first `import torch` to avoid error. There is a dependency. 
# setup(name='lltm_cpp',
#       ext_modules=[
#         cpp_extension.CppExtension('lltm_cpp', [
#             'lltm.cpp'
#             ])
#         ],
#       cmdclass={'build_ext': cpp_extension.BuildExtension}
# )