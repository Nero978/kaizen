#include "mnist.hpp"
#include "quantization.h"
#include "CNN.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {
uint32_t read_u32(std::istream &in) {
    unsigned char bytes[4];
    if (!in.read(reinterpret_cast<char *>(bytes), 4))
        throw std::runtime_error("Truncated MNIST IDX header");
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | bytes[3];
}
}

MnistBatch load_mnist(const std::string &directory, int count, int offset) {
    if (count <= 0 || offset < 0)
        throw std::runtime_error("Invalid MNIST batch size or offset");
    std::ifstream images(directory + "/train-images-idx3-ubyte", std::ios::binary);
    std::ifstream labels(directory + "/train-labels-idx1-ubyte", std::ios::binary);
    if (!images || !labels)
        throw std::runtime_error("Missing MNIST IDX files; run download-mnist.py first");
    const auto image_magic = read_u32(images), image_count = read_u32(images);
    const auto rows = read_u32(images), cols = read_u32(images);
    const auto label_magic = read_u32(labels), label_count = read_u32(labels);
    if (image_magic != 2051 || label_magic != 2049 || rows != 28 || cols != 28 ||
        image_count != 60000 || label_count != image_count ||
        uint64_t(offset) + uint64_t(count) > image_count)
        throw std::runtime_error("Invalid MNIST training headers or batch range");
    images.seekg(0, std::ios::end);
    labels.seekg(0, std::ios::end);
    if (images.tellg() != std::streampos(16 + image_count * 784) ||
        labels.tellg() != std::streampos(8 + label_count))
        throw std::runtime_error("Invalid MNIST IDX file lengths");
    images.seekg(16 + uint64_t(offset) * 784);
    labels.seekg(8 + offset);
    MnistBatch batch;
    batch.offset = offset;
    batch.images.resize(count, vector<vector<vector<F>>>(
        1, vector<vector<F>>(32, vector<F>(32, F_ZERO))));
    batch.flattened.resize(count, vector<F>(1024, F_ZERO));
    for (int i = 0; i < count; ++i) {
        unsigned char pixels[784], label;
        if (!images.read(reinterpret_cast<char *>(pixels), sizeof(pixels)) ||
            !labels.read(reinterpret_cast<char *>(&label), 1) || label > 9)
            throw std::runtime_error("Invalid MNIST sample or label");
        batch.labels.push_back(label);
        unsigned int pixel_sum = 0, nonzero = 0;
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                const auto pixel = pixels[y * 28 + x];
                const F value = quantize(float(pixel) / 255.0f);
                batch.images[i][0][y + 2][x + 2] = value;
                batch.flattened[i][(y + 2) * 32 + x + 2] = value;
                pixel_sum += pixel;
                nonzero += pixel != 0;
            }
        }
        printf("MNIST sample index=%d label=%u pixel_sum=%u nonzero_pixels=%u\n",
               offset + i, unsigned(label), pixel_sum, nonzero);
    }
    printf("Dataset: MNIST train; input=32x32x1; padding=2; normalization=pixel/255; Q=%d\n", Q);
    return batch;
}

vector<vector<F>> mnist_loss_gradient(const vector<vector<F>> &logits,
                                     const vector<int> &labels) {
    if (logits.size() != labels.size() || logits.empty())
        throw std::runtime_error("MNIST logit batch shape mismatch");
    vector<vector<F>> gradient(logits.size(), vector<F>(16, F_ZERO));
    double total_loss = 0;
    int correct = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i].size() != 16 || labels[i] < 0 || labels[i] > 9)
            throw std::runtime_error("MNIST classification head shape mismatch");
        double scores[10], probabilities[10];
        // The final dense output multiplies Q15 activations and Q15 weights.
        // Convert signed field residues directly to doubles to retain Q30 precision.
        for (int j = 0; j < 10; ++j) {
            F value = logits[i][j];
            const bool negative = value.getBit(60);
            if (negative) value = F_ZERO - value;
            scores[j] = double(value.toint128()) / double(1ULL << (2 * Q));
            if (negative) scores[j] = -scores[j];
        }
        const double maximum = *std::max_element(scores, scores + 10);
        double denominator = 0;
        for (int j = 0; j < 10; ++j) denominator += std::exp(scores[j] - maximum);
        const double loss = maximum + std::log(denominator) - scores[labels[i]];
        if (!std::isfinite(loss)) throw std::runtime_error("Non-finite MNIST loss");
        const int prediction = int(std::max_element(scores, scores + 10) - scores);
        total_loss += loss;
        correct += prediction == labels[i];
        for (int j = 0; j < 10; ++j) {
            probabilities[j] = std::exp(scores[j] - maximum) / denominator;
            gradient[i][j] = quantize(float(probabilities[j] - (j == labels[i] ? 1.0 : 0.0)));
        }
        printf("MNIST classification sample=%zu label=%d predicted=%d cross_entropy=%.9f\n",
               i, labels[i], prediction, loss);
        printf("MNIST output_gradient_q15:");
        for (int j = 0; j < 16; ++j) printf(" %.0f", double(dequantize(gradient[i][j], 1)) * (1 << Q));
        printf("\n");
    }
    printf("MNIST mean_cross_entropy=%.9f correct=%d/%zu; head=10 classes+6 masked slots\n",
           total_loss / labels.size(), correct, labels.size());
    printf("Loss/softmax computed on host; existing proof checks downstream arithmetic only.\n");
    return gradient;
}

void save_mnist_step(const convolutional_network &net, const MnistBatch &batch,
                     const std::string &path) {
    if (net.convolutions.empty() || net.convolutions.front().X.size() != batch.flattened.size())
        throw std::runtime_error("Network input does not match the loaded MNIST batch");
    // The actual convolution/proof input is stored with both image axes reversed.
    // Upstream real_X is an unused debug field and is not populated.
    for (size_t i = 0; i < batch.flattened.size(); ++i) {
        const auto &actual = net.convolutions.front().X[i];
        if (actual.size() != batch.flattened[i].size())
            throw std::runtime_error("Network input dimensions do not match MNIST");
        for (size_t j = 0; j < actual.size(); ++j)
            if (actual[actual.size() - 1 - j] != batch.flattened[i][j])
                throw std::runtime_error("Network convolution did not use the real MNIST input");
    }
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write MNIST SGD checkpoint: " + path);
    const auto signed_value = [](F value) -> long long {
        if (value.getBit(60)) return -static_cast<long long>((F_ZERO - value).toint128());
        return static_cast<long long>(value.toint128());
    };
    size_t changed = 0, count = 0;
    out << "{\"dataset\":\"MNIST train\",\"offset\":" << batch.offset
        << ",\"batch_size\":" << batch.labels.size()
        << ",\"Q\":" << Q << ",\"learning_rate_q15\":" << signed_value(quantize(0.01f))
        << ",\"scope\":\"one host SGD update; not an end-to-end training proof\",\"labels\":[";
    for (size_t i = 0; i < batch.labels.size(); ++i) {
        if (i) out << ',';
        out << batch.labels[i];
    }
    out << "],\"input_q15\":[";
    const auto write_matrix = [&](const vector<vector<F>> &matrix) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (i) out << ',';
            out << '[';
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                if (j) out << ',';
                out << signed_value(matrix[i][j]);
            }
            out << ']';
        }
    };
    write_matrix(batch.flattened);
    out << "],\"logits_q30\":[";
    write_matrix(net.fully_connected.back().Z_new);
    out << "],\"output_gradient_q15\":[";
    write_matrix(net.fully_connected_backprop.front().dx_input);
    out << "],\"layers\":[";
    bool first_layer = true;
    const auto write_layer = [&](const std::string &type, size_t index,
                                 const vector<int> &shape, const vector<F> &before,
                                 const vector<F> &delta) {
        if (before.size() != delta.size()) throw std::runtime_error("SGD shape mismatch");
        if (!first_layer) out << ',';
        first_layer = false;
        out << "{\"type\":\"" << type << "\",\"index\":" << index << ",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) { if (i) out << ','; out << shape[i]; }
        out << "],\"before_q15\":[";
        for (size_t i = 0; i < before.size(); ++i) { if (i) out << ','; out << signed_value(before[i]); }
        out << "],\"update_q15\":[";
        for (size_t i = 0; i < delta.size(); ++i) { if (i) out << ','; out << signed_value(delta[i]); }
        out << "],\"after_q15\":[";
        for (size_t i = 0; i < before.size(); ++i) {
            if (i) out << ',';
            out << signed_value(before[i] - delta[i]);
            changed += delta[i] != F_ZERO;
            ++count;
        }
        out << "]}";
    };
    for (size_t i = 0; i < net.Filters.size(); ++i) {
        const auto &weights = net.Filters[i];
        const auto &gradient = net.convolutions_backprop.at(net.Filters.size() - 1 - i);
        vector<F> before, delta;
        for (size_t o = 0; o < weights.size(); ++o)
            for (size_t c = 0; c < weights[o].size(); ++c)
                for (size_t y = 0; y < weights[o][c].size(); ++y)
                    for (size_t x = 0; x < weights[o][c][y].size(); ++x) {
                        before.push_back(weights[o][c][y][x]);
                        delta.push_back(gradient.dw.at(o * weights[o].size() + c).at(y * gradient.kernel_w + x));
                    }
        write_layer("conv", i,
            {int(weights.size()), int(weights[0].size()), int(weights[0][0].size()), int(weights[0][0][0].size())}, before, delta);
    }
    for (size_t i = 0; i < net.Weights.size(); ++i) {
        const auto &weights = net.Weights[i];
        const auto &gradient = net.fully_connected_backprop.at(net.Weights.size() - 1 - i);
        vector<F> before, delta;
        for (size_t o = 0; o < weights.size(); ++o)
            for (size_t j = 0; j < weights[o].size(); ++j) {
                before.push_back(weights[o][j]);
                delta.push_back(gradient.dw.at(o).at(j));
            }
        write_layer("dense", i, {int(weights.size()), int(weights[0].size())}, before, delta);
    }
    out << "],\"parameters\":" << count << ",\"changed_parameters\":" << changed << "}\n";
    out.close();
    if (!out) throw std::runtime_error("Failed to write SGD checkpoint");
    printf("MNIST SGD checkpoint: parameters=%zu changed=%zu file=%s\n", count, changed, path.c_str());
}
