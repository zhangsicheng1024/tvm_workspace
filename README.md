## container
```shell
docker pull zhangsicheng1024/ubuntu18.04-cuda11.0-llvm10
docker run -it --privileged=true -d --gpus all -v xxx/tvm_workspace:/tvm_workspace zhangsicheng1024/ubuntu18.04-cuda11.0-llvm10:latest /bin/bash
```

## packages
需要的包在容器里应该是装好了的，不过如果有缺的可以按照这个再安装一次
```shell
apt-get update
apt-get install wget
wget https://apt.kitware.com/kitware-archive.sh
chmod +x kitware-archive.sh
./kitware-archive.sh

# basic
apt-get install -y python3 python3-dev python3-setuptools python3-pip gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev vim git tmux

# llvm
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
cp -r clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/* /usr/

# python
pip3 install --user numpy decorator attrs typing-extensions psutil scipy tornado psutil 'xgboost>=1.1.0' cloudpickle pytest
```

## run code
### env
```shell
export TVM_HOME=xxx/apache-tvm-src-v0.10.0
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

### tvm src  
tvm源码主要是apache-tvm-src-v0.10.0  
仓库里应该有两次commit，1 init code from apache-tvm v0.10.0是修改前tvm的code，2 low energy kernel search code是加入了我们自己修改的内容  
tvm src build 流程：按前面流程安装好package并设置好环境变量后这里应该可以直接build
```shell
cd apache-tvm-src-v0.10.0/build
cp ../_build/config.cmake . # 这是我们之前用的config
# or
cp ../cmake/config.cmake . # tvm官方config，可以用这个从头修改

cmake ..
make -j8
# 生成 libtvm.so libtvm_runtime.so两个库就build好了
```  
如果有问题可以参考  
https://tvm.apache.org/docs/install/from_source.html#install-from-source

### 实验基本流程
```shell
cd xxx/tvm_workspace/tvm_test/matmul

# 修改 line26 method(生成的log的名字), search_count(搜索iteration), MNK
# 用这个脚本进行搜索，结果是一个 xxx.json
python matmul_energy.py

# 修改 line9 name，对应json的名字
# 用这个脚本可以跑完从json生成kernel并在gpu上运行测量出energy的流程
python energy_measure.py
```