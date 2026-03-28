#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "tinyvit_hier.hpp"

using namespace marr;


int main() {
    torch::manual_seed(42);
    torch::set_num_threads(1);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    Marr::TinyViTConfig cfg;
    cfg.image_size = 32;
    cfg.in_channels = 1;
    cfg.num_classes = 4;
    cfg.embed_dims = {32, 64, 96, 128};
    cfg.depths = {2, 2, 2, 1};
    cfg.num_heads = {1, 2, 3, 4};
    cfg.window_size = 4;
    cfg.mlp_ratio = 4;

    constexpr int64_t kTrainSize = 256;
    constexpr int64_t kValSize = 128;
    constexpr int64_t kBatchSize = 32;
    constexpr int64_t kEpochs = 8;

    auto [train_x_cpu, train_y_cpu] =
        tinyvit::make_toy_dataset(kTrainSize, cfg.num_classes, cfg.image_size);
    auto [val_x_cpu, val_y_cpu] =
        tinyvit::make_toy_dataset(kValSize, cfg.num_classes, cfg.image_size);

    auto model = Marr::TinyViTHierarchical(cfg);
    model->to(device);
    model->print_summary();

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(1e-4));

    for (int64_t epoch = 1; epoch <= kEpochs; ++epoch) {
        model->train();
        auto permutation = torch::randperm(kTrainSize, torch::kLong);
        auto train_x = train_x_cpu.index_select(0, permutation).to(device);
        auto train_y = train_y_cpu.index_select(0, permutation).to(device);

        double epoch_loss = 0.0;
        int64_t num_batches = 0;

        for (int64_t start = 0; start < kTrainSize; start += kBatchSize) {
        const int64_t end = std::min(start + kBatchSize, kTrainSize);
        auto xb = train_x.index({torch::indexing::Slice(start, end)});
        auto yb = train_y.index({torch::indexing::Slice(start, end)});

        optimizer.zero_grad();
        auto logits = model->forward(xb);
        auto loss = torch::nn::functional::cross_entropy(logits, yb);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();

        epoch_loss += loss.item<double>();
        ++num_batches;
        }

        model->eval();
        torch::NoGradGuard no_grad;
        auto val_logits = model->forward(val_x_cpu.to(device));
        auto val_acc = tinyvit::accuracy(val_logits.cpu(), val_y_cpu);

        std::cout << "Epoch " << std::setw(2) << epoch << " | loss=" << std::fixed
                << std::setprecision(4) << (epoch_loss / num_batches)
                << " | val_acc=" << std::setprecision(3) << val_acc * 100.0 << "%\n";
    }
    
    const std::string checkpoint = "tinyvit_hier_patterns.pt";
    torch::save(model, checkpoint);
    std::cout << "Saved checkpoint to: " << checkpoint << "\n";
    return 0;
}