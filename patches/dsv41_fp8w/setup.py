from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="dsv41_fp8w_ext",
      ext_modules=[CUDAExtension("dsv41_fp8w_ext", ["bind.cpp", "fp8w.cu"],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-gencode=arch=compute_80,code=sm_80"]})],
      cmdclass={"build_ext": BuildExtension})
