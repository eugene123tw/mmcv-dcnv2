import glob
import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    extensions = []
    ext_name = 'ext'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []
    extra_compile_args = {'cxx': []}

    define_macros += [('MMCV_WITH_CUDA', None)]
    cuda_args = os.getenv('MMCV_CUDA_ARGS')
    extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
    op_files = glob.glob('./src/pytorch/*')
    extension = CUDAExtension

    include_path = os.path.abspath('./src')
    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions


setup(
    name='mmcv',
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Authors',
    author_email='openmmlab@gmail.com',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False)
