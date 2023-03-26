#include <bits/stdc++.h>
#include <torch/torch.h>

#include "util.h"
#include "dcgan.h"

using namespace std;
using namespace torch;
using namespace torch::data;
using namespace torch::optim;

namespace
{

    const string path = "/scratch/workspaceblobstore/LocalUpload/User/sahuamrit/Repo/pytorch_cpp_frontend/data";
    const int64_t kBatchSize = 64;
    const int64_t kNoiseSize = 100;
    const int64_t kNumberOfEpochs = 30;
    const int64_t kCheckpointEvery = 100;
} // namespace

int main(int argc, char **argv)
{
    cout << "Training Started"
         << "\n";

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }

    datasets::MNIST dataset = datasets::MNIST(path);

    auto transform_dataset = compose(std::move(dataset), transforms::Normalize<Tensor>(0.5, 0.5), transforms::Stack<>());
    // std::move converts an lvalue to rvalue and allows moving instead of copying
    //
    // compose() can't use SourceType& dataset since the dataset.map(transform) is
    // rvalue

    // Load data
    int64_t batches_per_epoch =
        (ceil(transform_dataset.size().value() / static_cast<double>(kBatchSize)));

    auto data_loader = make_data_loader(
        std::move(transform_dataset),
        DataLoaderOptions().batch_size(kBatchSize).workers(2));

    // for (Example<> &batch : *data_loader)
    // {
    //     std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    //     for (int64_t i = 0; i < batch.data.size(0); ++i)
    //     {
    //         std::cout << batch.target[i].item<int64_t>() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Load Model

    auto generator = DCGANGenerator(kNoiseSize);
    auto discriminator = DCGANDiscriminator();

    generator->to(device);
    discriminator->to(device);

    // Initialise optimizer

    auto generator_optimizer = torch::optim::Adam(generator->parameters(), AdamOptions(5e-4));
    auto discriminator_optimizer = torch::optim::Adam(discriminator->parameters(), AdamOptions(2e-4));

    // Training Loop

    int64_t checkpoint_counter = 0;
    for (int64_t epoch = 0; epoch < kNumberOfEpochs; epoch++)
    {
        int64_t batch_idx = 0;
        for (auto &batch : *data_loader)
        {
            auto batch_size = batch.data.size(0);
            auto real_img = batch.data.to(device);

            auto input_noise = torch::randn({batch_size, kNoiseSize, 1, 1}, device);

            // prepare labels
            auto lbl_real = torch::empty(batch_size, device).uniform_(0.8, 1);
            auto lbl_fake = torch::zeros(batch_size, device);

            // Train discriminator
            discriminator_optimizer.zero_grad();

            auto fake_img = generator->forward(input_noise);
            auto d_pred_fake = discriminator->forward(fake_img.detach());
            auto d_loss_fake = torch::binary_cross_entropy(d_pred_fake, lbl_fake);
            d_loss_fake.backward();

            auto d_pred_real = discriminator->forward(real_img);
            auto d_loss_real = torch::binary_cross_entropy(d_pred_real, lbl_real);
            d_loss_real.backward();

            discriminator_optimizer.step();
            auto d_loss = d_loss_real + d_loss_fake;

            // Train generator
            generator_optimizer.zero_grad();

            auto gen_img = generator->forward(input_noise);
            auto d_pred_gen = discriminator->forward(gen_img);
            auto g_loss = torch::binary_cross_entropy(d_pred_gen, lbl_real);
            g_loss.backward();

            generator_optimizer.step();

            printf(
                "[%2ld/%2ld][%ld/%ld] D_loss: %.4f | G_loss: %.4f \n",
                epoch,
                kNumberOfEpochs,
                ++batch_idx,
                batches_per_epoch,
                d_loss.item<float>(),
                g_loss.item<float>());

            if (batch_idx % kCheckpointEvery == 0)
            {
                // Checkpoint the model and optimizer state.
                torch::save(generator, "generator-checkpoint.pt");
                torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "discriminator-checkpoint.pt");
                torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
                // Sample the generator and save the images.
                torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
                torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
                cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }
        }
    }
    cout << "Training complete!"
         << "\n";
}
