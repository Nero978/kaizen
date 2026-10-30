#pragma once
#include "config_pc.hpp"
#include <string>

struct MnistBatch {
    vector<vector<vector<vector<F>>>> images;
    vector<vector<F>> flattened;
    vector<int> labels;
    int offset;
};

MnistBatch load_mnist(const std::string &directory, int count, int offset);
vector<vector<F>> mnist_loss_gradient(const vector<vector<F>> &logits,
                                     const vector<int> &labels);
struct convolutional_network;
void save_mnist_step(const convolutional_network &net, const MnistBatch &batch,
                     const std::string &path);
