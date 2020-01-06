from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name='fusion_cuda',
  ext_modules=[
    CUDAExtension('fusion_cuda', [
      'fusion_cuda.cpp',
      'fusion_cuda_kernel.cu',
    ]),
  ],
  cmdclass={
    'build_ext': BuildExtension
  })