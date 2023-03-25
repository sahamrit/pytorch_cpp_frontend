#include <bits/stdc++.h>
#include <torch/torch.h>

using namespace std;
using namespace torch;

struct DCGANGeneratorImpl : nn::Module
{

  // Constructor
  DCGANGeneratorImpl(int kNoiseSize) : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                                                 .bias(false)),
                                       batch_norm1(nn::BatchNorm2dOptions(256)),
                                       conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                                                 .stride(2)
                                                 .padding(1)
                                                 .bias(false)),
                                       batch_norm2(nn::BatchNorm2dOptions(128)),
                                       conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                                                 .stride(2)
                                                 .padding(1)
                                                 .bias(false)),
                                       batch_norm3(nn::BatchNorm2dOptions(64)),
                                       conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                                                 .stride(2)
                                                 .padding(1)
                                                 .bias(false))
  {
    // register all the sub modules
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
  }

  // Declaration of forward method
  torch::Tensor forward(torch::Tensor x);

  // Initialized by C++ constructor List
  // Doesn't require having a MyClass() constructor definition
  nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);

struct DCGANDiscriminatorImpl : nn::Module
{
  DCGANDiscriminatorImpl() : conv1(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
                                                     l_relu1(nn::LeakyReLUOptions().negative_slope(0.2)),
                                                     conv2(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
                                                     batch_norm2(nn::BatchNorm2dOptions(128)),
                                                     l_relu2(nn::LeakyReLUOptions().negative_slope(0.2)),
                                                     conv3(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
                                                     batch_norm3(nn::BatchNorm2dOptions(256)),
                                                     l_relu3(nn::LeakyReLUOptions().negative_slope(0.2)),
                                                     conv4(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false))
  {
    // register all the sub modules
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);

    register_module("l_relu1", l_relu1);
    register_module("l_relu2", l_relu2);
    register_module("l_relu3", l_relu3);
  }

  // Declaration of forward method
  torch::Tensor forward(torch::Tensor x);

  nn::Conv2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm2, batch_norm3;
  nn::LeakyReLU l_relu1, l_relu2, l_relu3;
};

TORCH_MODULE(DCGANDiscriminator);