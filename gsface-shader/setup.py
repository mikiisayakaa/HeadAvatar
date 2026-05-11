from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="gsface_shader",
    #packages=['gsface_shader'],
    ext_modules=[
        CUDAExtension(
            name="_gsface_shader",
            sources=[
                "etc.cu",
                "bruteforce_shader.cu",
                "bindings.cpp"
            ],
            # extra_link_args=['-L/opt/miniconda/envs/testenv/x86_64-conda-linux-gnu/sysroot/lib64', '-L/usr/lib/x86_64-linux-gnu'],
            # extra_compile_args={
            #     "cxx": ["-O0"],
            #     "nvcc": ["-O0", "-G", "-lineinfo"]
            # }
         )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(verbose=True)
    }
)