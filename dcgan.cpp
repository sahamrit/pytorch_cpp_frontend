#include <bits/stdc++.h>
#include <torch/torch.h>

#include "dcgan.h"

using namespace std;
using namespace torch;

torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x)
{
  x = torch::relu(batch_norm1(conv1(x)));
  x = torch::relu(batch_norm2(conv2(x)));
  x = torch::relu(batch_norm3(conv3(x)));
  x = torch::tanh(conv4(x));
  return x;
};


torch::Tensor DCGANDiscriminatorImpl::forward(torch::Tensor x)
{
  x = l_relu1(conv1(x));
  x = l_relu2(batch_norm2((conv2(x))));
  x = l_relu3(batch_norm3((conv3(x))));
  x = torch::sigmoid(conv4(x));
  return x;
};
