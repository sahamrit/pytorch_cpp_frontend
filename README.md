# pytorch_cpp_frontend
This repository contains learning exercises for pytorch cpp frontend

### Requirements

1. Installation of LibTorch. Follow instructions in this [pytorch tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html#) under `Writing a Basic Application section`
2. C++ proficiency and knowledge of C++ [Pytorch Frotend](https://pytorch.org/tutorials/advanced/cpp_frontend.html#)
3. Intel MKL Installation 
```
sudo apt-get -y install intel-mkl
```

### Build

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..
cmake --build . --config Release
```

### Run

```
// Inside build directory
./currentExecutable
```

### Display generated images

```
// Inside pytorch_cpp_fronted

python display.py -i build/dcgan-sample-200.pt 
```

![alt text](./out.png)



## Additional Checkpoint Branches

### Checkpoint 1: Implement chain of transformations in C++

Solution : In branch [variadic_templates](https://github.com/sahamrit/pytorch_cpp_frontend/tree/variadic_templates)

### Checkpoint 2: DCGAN generating MNIST Images

Solution : In branch [dcgan_v1](https://github.com/sahamrit/pytorch_cpp_frontend/tree/dcgan_v1)
