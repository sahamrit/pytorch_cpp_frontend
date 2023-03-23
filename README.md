# pytorch_cpp_frontend
This repository contains learning exercises for pytorch cpp frontend.

## Exercise : Implement chain of transformations in C++

### Problem Statement

Since Python is a dynamically typed language, we can do things like - create a dict with multiple types of keys, pass variable list of parameters etc. All of this shows the ease and power of dynamically typed language.

In python if you have a dataset, it is super easy to do 

```
compose = transforms.Compose([vision.Decode(), vision.RandomCrop(512)])
image_folder_dataset = image_folder_dataset.map(operations=compose)
```

If you dive into details the map function can take a list of transformations and do a composition like `f(g(h(x)))`. It is easy to see that if the return type will depend on the length of transforms passed.

### Solution

While learning [Pytorch C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_frontend.html) I came across 

```
auto dataset = torch::data::datasets::MNIST("./mnist")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
```

I was wandering if it is possible to bring python like interface here out of curiosity. Something like

```
auto transform_dataset = compose(std::move(dataset), transforms::Normalize<Tensor>(0.5, 0.5), transforms::Stack<>());
```

#### Thought Process

C++ is statically typed. How will it be possible to create a function whose return type depends on length of parameter list passed to function.

Can we somehow use C++ template programming. We have two problems here. During writing the function definition how do I write the return type, which is of variable length. Something like 
```
MapDataset<MapDataset<MapDataset,TransformType1>,TransformType2>
```
. Here the TransformTypes are unknown to the function and also the length of this chain is unknown. How do I pass a variable length argument list of various types.

C++ offers solutions to both which I found after many attempts :). Using `auto` return type lets the compiler do [automatic return type-deduction](https://www.geeksforgeeks.org/return-type-deduction-in-c14-with-examples/) during compile time. C++ also offers [variadic templates](https://en.cppreference.com/w/cpp/language/parameter_pack) which allows you to pass variable length arguments of different types. Using [template meta programming](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Templates/Template_Meta-Programming) the compiler at compile time is able to resolve the types of this call 

```
auto transform_dataset = compose(std::move(dataset), transforms::Normalize<Tensor>(0.5, 0.5), transforms::Stack<>());
```

Note this is not truly dynamic. Had we passed this list of transforms via terminal this couldn't have been handled by C++ and requires dynamically typed language as python

[Solution Code](util.h)

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
./dcgan
```


