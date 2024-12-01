#ifndef PROJECT2_A_H
#define PROJECT2_A_H

// Include header file
#include "project2_a_basics.h"
#include "dense_linear_algebra.h"

// Import necessary libraries
#include <cmath>
#include <vector>
#include <cassert>

// Import LinearAlgebra namespace
using namespace BasicDenseLinearAlgebra;

class Layer{
public:
    Layer() = delete;
    Layer(int input_size, int output_size, double l_rate);
    DoubleVector feed_forward(const DoubleVector &input) const;
    int get_input_dim() const { return input_dim; }

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
Layer::Layer(int input_size, int output_size, double l_rate)
    : weights(output_size, input_size), bias(output_size) {
    assert(input_size > 0);
    assert(output_size > 0);

    output_dim = output_size;
    input_dim = input_size;
    eta = l_rate;

    weights = DoubleMatrix(output_size, input_size);
    bias = DoubleVector(output_size);
};

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



#endif // PROJECT2_A_H