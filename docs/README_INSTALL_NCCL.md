### Debian/Ubuntu Installation Instructions

1. install deps for packaging
```
sudo apt update
sudo apt install -y build-essential devscripts debhelper fakeroot
```

2. get source
```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
# optional: checkout a specific release tag
# git checkout v2.xx.x-1
```

3. build the deb package
```
make -j pkg.debian.build
```

4. install the package(s)
```
ls build/pkg/deb/
sudo dpkg -i build/pkg/deb/*.deb
sudo ldconfig
```

The pkg.debian.build target and output path are documented in the NCCL repo.

### RedHat/CentOS Installation Instructions

1. deps for packaging
```
sudo yum install -y rpm-build rpmdevtools  # (or dnf on newer distros)
```

2. get source
```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
# optional: git checkout v2.xx.x-1
```

3. build the rpm package
```
make -j pkg.redhat.build
```

4. install the package(s)
```
ls build/pkg/rpm/
sudo rpm -Uvh build/pkg/rpm/*.rpm
sudo ldconfig
```

### Installation without sudo/root
1. get source
```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
# optional: git checkout v2.xx.x-1
```

2. build NCCL (set CUDA\_HOME if CUDA isn't in /usr/local/cuda)
```
make -j src.build CUDA_HOME=/usr/local/cuda
```

3. install headers+libs to a custom prefix
```
make install PREFIX=<your_install_path>
```

4. make runtime linker find it
```
export LD_LIBRARY_PATH=<your_install_path>/nccl/lib:$LD_LIBRARY_PATH
```
