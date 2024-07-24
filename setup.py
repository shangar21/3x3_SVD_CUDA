from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='svd_extension',
    ext_modules=[
        CUDAExtension(
            name = 'svd_extension', 
            sources = [
                'svd_extension.cpp',
                './svd3x3/svd3x3/svd3_cuda.cu'
            ],
            extra_compile_args = {
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

