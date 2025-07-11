"""Setup configuration for Python packaging."""
# pylint: disable = deprecated-module, exec-used
import os
import sys
import sysconfig
import platform
import glob
from pathlib import Path
from setuptools import setup
from setuptools.command.install import install
from torch.utils import cpp_extension

def parse_library_names(libdir):
    library_names = []
    for root, _, files in os.walk(libdir):
        for file in files:
            if file.endswith((".a", ".so", ".dylib")):
                file_name = os.path.basename(file)
                library_names.append(file_name[3:].rsplit(".", 1)[0])

    # add system netcdf library
    library_names.extend(['netcdf'])

    # move current library name to first
    current = [item for item in library_names if item.startswith('snap')]
    other = [item for item in library_names if not item.startswith('snap')]
    return current + other

site_dir = sysconfig.get_paths()["purelib"]

current_dir = os.getenv("WORKSPACE", Path().absolute())
include_dirs = [
    f"{current_dir}",
    f"{current_dir}/build",
    f"{current_dir}/build/_deps/fmt-src/include",
    f'{current_dir}/build/_deps/yaml-cpp-src/include',
    f"{site_dir}/kintera",
]

# add homebrew directories if on MacOS
lib_dirs = [f"{current_dir}/build/lib"]
if platform.system() == 'Darwin':
    lib_dirs.extend(['/opt/homebrew/lib'])
else:
    lib_dirs.extend(['/lib64/', '/usr/lib/x86_64-linux-gnu/'])

libraries = parse_library_names(f"{current_dir}/build/lib")

if sys.platform == "darwin":
    extra_link_args = [
        "-Wl,-rpath,@loader_path/lib",
        "-Wl,-rpath,@loader_path/../torch/lib",
        "-Wl,-rpath,@loader_path/../pydisort/lib",
        "-Wl,-rpath,@loader_path/../pyharp/lib",
        "-Wl,-rpath,@loader_path/../kintera/lib",
    ]
else:
    extra_link_args = [
        "-Wl,-rpath,$ORIGIN/lib",
        "-Wl,-rpath,$ORIGIN/../torch/lib",
        "-Wl,-rpath,$ORIGIN/../pydisort/lib",
        "-Wl,-rpath,$ORIGIN/../pyharp/lib",
        "-Wl,-rpath,$ORIGIN/../kintera/lib",
    ]

ext_module = cpp_extension.CppExtension(
    name='snapy.snapy',
    sources=glob.glob('python/csrc/*.cpp'),
    include_dirs=include_dirs,
    library_dirs=lib_dirs,
    libraries=libraries,
    extra_compile_args=['-Wno-attributes'],
    extra_link_args=extra_link_args,
    )

setup(
    package_dir={"snapy": "python"},
    ext_modules=[ext_module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
