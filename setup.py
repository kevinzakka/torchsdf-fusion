from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
  name='fusion_cpp',
  ext_modules=[
    CppExtension('fusion_cpp', ['fusion.cpp']),
  ],
  cmdclass={
    'build_ext': BuildExtension
  })