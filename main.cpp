#include <bits/stdc++.h>
#include <torch/torch.h>

#include "util.h"

using namespace std;
using namespace torch;
using namespace torch::data;

namespace
{

    const string path = "/scratch/workspaceblobstore/LocalUpload/User/sahuamrit/Repo/pytorch_cpp_frontend/data";
    const int kBatchSize = 64;

} // namespace

int main(int argc, char **argv)
{

    datasets::MNIST dataset = datasets::MNIST(path);

    auto transform_dataset = compose(std::move(dataset), transforms::Normalize<Tensor>(0.5, 0.5), transforms::Stack<>());
    // std::move converts an lvalue to rvalue and allows moving instead of copying
    //
    // compose() can't use SourceType& dataset since the dataset.map(transform) is
    // rvalue

    auto data_loader = make_data_loader(
        std::move(transform_dataset),
        DataLoaderOptions().batch_size(kBatchSize).workers(2));

    for (Example<> &batch : *data_loader)
    {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i)
        {
            std::cout << batch.target[i].item<int64_t>() << " ";
        }
        std::cout << std::endl;
    }
}
