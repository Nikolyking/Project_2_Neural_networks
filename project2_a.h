#ifndef PROJECT2_A_H
#define PROJECT2_A_H

// Include header file
#include "project2_a_basics.h"
#include "dense_linear_algebra.h"

// Import necessary libraries
#include <cmath>
#include <vector>
#include <cassert>
#include <fstream>
#include <string>

// Import LinearAlgebra namespace
using namespace BasicDenseLinearAlgebra;

class Layer{
public:
    Layer() = delete;
    Layer(int input_size, int output_size, double l_rate);
    DoubleVector feed_forward(const DoubleVector &input) const;
    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return input_dim; }
    const DoubleMatrix& get_weights() const { return weights; }
    const DoubleVector& get_bias() const { return bias; }
    void set_weights(const DoubleMatrix& new_weights) { weights = new_weights; }
    void set_bias(const DoubleVector& new_bias) { bias = new_bias; }

private:
    DoubleMatrix weights;
    DoubleVector bias;
    int output_dim;
    int input_dim;
    double eta;
};

// Constructor for Layer class
Layer::Layer(int input_size, int output_size, double l_rate)
    : weights(output_size, input_size), bias(output_size) {
    assert(input_size > 0);
    assert(output_size > 0);

    output_dim = output_size;
    input_dim = input_size;
    eta = l_rate;

    weights = DoubleMatrix(output_size, input_size);
    bias = DoubleVector(output_size);
}

DoubleVector Layer::feed_forward(const DoubleVector &input) const {
    assert(input.n() == input_dim);

    DoubleVector output(output_dim);

    for (unsigned i = 0; i < output_dim; ++i) {
        output[i] = bias[i];
        for (unsigned j = 0; j < input_dim; ++j) {
            output[i] += weights(i, j) * input[j];
        }
        output[i] = std::tanh(output[i]); 
    }

    return output;
};

class NeuralNetwork : public NeuralNetworkBasis{
    public:
    // Function that evaluates the feed-forward algorithm running
    // through the entire network, for input x. Feed it through the 
    // non-input-layers and return the output from the final layer.
    // Overrides base class function
    void feed_forward(const DoubleVector& input, DoubleVector& output) const override;

    // Function to calculate the cost for given input and target output
    // Overrides base class function
    double cost(const DoubleVector& input, const DoubleVector& target_output) const override;

    double cost_for_training_data(const std::vector<std::pair<DoubleVector,
    DoubleVector>> training_data) const override;

    void write_parameters_to_disk(const std::string& filename) const override;

    void read_parameters_from_disk(const std::string& filename) override;

    private:
        std::vector<Layer> layers;
};

// Definition of the feed_forward function
void NeuralNetwork::feed_forward(const DoubleVector& input, DoubleVector& output) const {
    assert(input.n() == layers.front().get_input_dim());

    DoubleVector current_input = input;

    for (const Layer& layer : layers) {
        current_input = layer.feed_forward(current_input);
    }

    output = current_input;
}

// Definition of the cost function
double NeuralNetwork::cost(const DoubleVector& input, const DoubleVector& target_output) const {
    DoubleVector output;
    feed_forward(input, output);

    assert(output.n() == target_output.n());

    double cost = 0.0;
    for (unsigned i = 0; i < output.n(); ++i) {
        double diff = output[i] - target_output[i];
        cost += diff * diff;
    }

    return cost / 2.0;
}

double NeuralNetwork::cost_for_training_data(const std::vector<std::pair<DoubleVector,
    DoubleVector>> training_data) const {
        double total_cost = 0;
        
        for (unsigned i = 0; i < training_data.size(); i++){
            const DoubleVector& input = training_data[i].first;
            const DoubleVector& target_output = training_data[i].second;
            total_cost += cost(input, target_output);
        }

        return total_cost / training_data.size();
    }

// Definition of the write_parameters_to_disk function
void NeuralNetwork::write_parameters_to_disk(const std::string& filename) const {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Unable to open file for writing");
    }

    for (const Layer& layer : layers) {
        outfile << "TanhActivationFunction" << std::endl;
        outfile << layer.get_input_dim() << " " << layer.get_output_dim() << std::endl;

        const DoubleVector& bias = layer.get_bias();
        for (unsigned i = 0; i < bias.n(); ++i) {
            outfile << i << " " << bias[i] << std::endl;
        }

        const DoubleMatrix& weights = layer.get_weights();
        for (unsigned i = 0; i < weights.n(); ++i) {
            for (unsigned j = 0; j < weights.m(); ++j) {
                outfile << i << " " << j << " " << weights(i, j) << std::endl;
            }
        }
    }

    outfile.close();
}

// Definition of the read_parameters_from_disk function
void NeuralNetwork::read_parameters_from_disk(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open file for reading");
    }

    layers.clear();

    std::string activation_function;
    while (infile >> activation_function) {
        if (activation_function != "TanhActivationFunction") {
            throw std::runtime_error("Unsupported activation function");
        }

        int input_dim, output_dim;
        infile >> input_dim >> output_dim;

        Layer layer(input_dim, output_dim, 0.0); // Learning rate is not needed here

        DoubleVector bias(output_dim);
        for (unsigned i = 0; i < output_dim; ++i) {
            int index;
            infile >> index >> bias[i];
        }
        layer.set_bias(bias);

        DoubleMatrix weights(output_dim, input_dim);
        for (unsigned i = 0; i < output_dim; ++i) {
            for (unsigned j = 0; j < input_dim; ++j) {
                int row, col;
                infile >> row >> col >> weights(i, j);
            }
        }
        layer.set_weights(weights);

        layers.push_back(layer);
    }

    infile.close();
}

#endif // PROJECT2_A_H