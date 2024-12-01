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
#include <random>

// Import LinearAlgebra namespace
using namespace BasicDenseLinearAlgebra;

class Layer{
public:
    Layer() = delete;
    Layer(int input_size, int output_size, double l_rate);

    DoubleVector feed_forward(const DoubleVector& input) const;
    void update_parameters(const DoubleMatrix& weight_gradients, const DoubleVector& bias_gradients, double learning_rate);
    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
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
    DoubleVector last_input;
    DoubleVector last_output;
};

// Constructor for Layer class
Layer::Layer(int input_size, int output_size, double l_rate)
    : weights(output_size, input_size), bias(output_size) {
    assert(input_size > 0);
    assert(output_size > 0);

    output_dim = output_size;
    input_dim = input_size;
    eta = l_rate;

    // Initialize weights and biases with small random values
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (unsigned i = 0; i < output_dim; ++i) {
        bias[i] = dist(rng);
        for (unsigned j = 0; j < input_dim; ++j) {
            weights(i, j) = dist(rng);
        }
    }
}

DoubleVector Layer::feed_forward(const DoubleVector& input) const {
    assert(input.n() == input_dim);

    DoubleVector output(output_dim);

    for (unsigned i = 0; i < output_dim; ++i) {
        double sum = bias[i];
        for (unsigned j = 0; j < input_dim; ++j) {
            output[i] += weights(i, j) * input[j];
        }
        output[i] = std::tanh(sum); 
    }

    return output;
};

void Layer::update_parameters(const DoubleMatrix& weight_gradients, const DoubleVector& bias_gradients, double learning_rate) {
    for (unsigned i = 0; i < output_dim; ++i) {
        bias[i] -= learning_rate * bias_gradients[i];
        for (unsigned j = 0; j < input_dim; ++j) {
            weights(i, j) -= learning_rate * weight_gradients(i, j);
        }
    }
}

class NeuralNetwork : public NeuralNetworkBasis{
    public:
    NeuralNetwork(unsigned int n_input, const std::vector<std::pair<unsigned int, ActivationFunction*>>& non_input_layer);

    // Function that evaluates the feed-forward algorithm running
    // through the entire network, for input x. Feed it through the 
    // non-input-layers and return the output from the final layer.
    // Overrides base class function
    void feed_forward(const DoubleVector& input, DoubleVector& output) const override;

    // Function to calculate the cost for given input and target output
    // Overrides base class function
    double cost(const DoubleVector& input, const DoubleVector& target_output) const override;

    // Function to calculate the cost for the entire training data
    // Overrides base class function
    double cost_for_training_data(std::vector<std::pair<DoubleVector,
    DoubleVector>> training_data) const override;

    // Function to write parameters to disk
    // Overrides base class function
    void write_parameters_to_disk(const std::string& filename) const override;

    // Function to read parameters from disk
    // Overrides base class function
    void read_parameters_from_disk(const std::string& filename) override;

    // Function to train the network
    // Overrides base class function
    void train(
    const std::vector<std::pair<DoubleVector,DoubleVector>>& training_data,
    const double& learning_rate,
    const double& tol_training,
    const unsigned& max_iter,
    const std::string& convergence_history_file_name="") override;

    private:
        std::vector<Layer> layers;

    // Helper functions for training
    void compute_gradients_finite_difference(const DoubleVector& input, const DoubleVector& target_output,
                                             std::vector<DoubleMatrix>& weight_gradients, std::vector<DoubleVector>& bias_gradients);
    void backpropagate(const DoubleVector& input, const DoubleVector& target_output, std::vector<DoubleMatrix>& weight_gradients, std::vector<DoubleVector>& bias_gradients);
};

NeuralNetwork::NeuralNetwork(unsigned int n_input, const std::vector<std::pair<unsigned int, ActivationFunction*>>& non_input_layer) {
    assert(non_input_layer.size() >= 1);
    double l_rate = 0.01;
    layers.emplace_back(Layer(n_input, non_input_layer[0].first, l_rate));
    for (size_t i = 1; i < non_input_layer.size(); ++i) {
        layers.emplace_back(Layer(non_input_layer[i - 1].first, non_input_layer[i].first, l_rate));
    }
}

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

double NeuralNetwork::cost_for_training_data(std::vector<std::pair<DoubleVector,
    DoubleVector>> training_data) const {
        double total_cost = 0.0;
        
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
        bias.output(outfile);

        const DoubleMatrix& weights = layer.get_weights();
        weights.output(outfile);
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

        Layer layer(input_dim, output_dim, 0.0);

        DoubleVector bias(output_dim);
        bias.read(infile);
        layer.set_bias(bias);

        DoubleMatrix weights(output_dim, input_dim);
        weights.read(infile);
        layer.set_weights(weights);

        layers.push_back(layer);
    }

    infile.close();
}

// Definition of the train function
void NeuralNetwork::train(const std::vector<std::pair<DoubleVector, DoubleVector>>& training_data,
                          const double& learning_rate,
                          const double& tol_training,
                          const unsigned& max_iter,
                          const std::string& convergence_history_file_name) {
    std::ofstream convergence_file;
    if (!convergence_history_file_name.empty()) {
        convergence_file.open(convergence_history_file_name);
        if (!convergence_file.is_open()) {
            throw std::runtime_error("Unable to open convergence history file for writing");
        }
    }

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, training_data.size() - 1);

    for (unsigned iter = 0; iter < max_iter; ++iter) {
        size_t idx = dist(rng);
        const DoubleVector& input = training_data[idx].first;
        const DoubleVector& target_output = training_data[idx].second;

        // Compute gradients using finite differencing
        std::vector<DoubleMatrix> fd_weight_gradients(layers.size());
        std::vector<DoubleVector> fd_bias_gradients(layers.size());
        compute_gradients_finite_difference(input, target_output, fd_weight_gradients, fd_bias_gradients);

        // Compute gradients using backpropagation
        std::vector<DoubleMatrix> bp_weight_gradients(layers.size());
        std::vector<DoubleVector> bp_bias_gradients(layers.size());
        backpropagate(input, target_output, bp_weight_gradients, bp_bias_gradients);

        // Compare gradients
        double max_diff = 0.0;
        for (size_t l = 0; l < layers.size(); ++l) {
            for (unsigned i = 0; i < layers[l].get_output_dim(); ++i) {
                double bias_diff = std::abs(fd_bias_gradients[l][i] - bp_bias_gradients[l][i]);
                if (bias_diff > max_diff) {
                    max_diff = bias_diff;
                }
                for (unsigned j = 0; j < layers[l].get_input_dim(); ++j) {
                    double weight_diff = std::abs(fd_weight_gradients[l](i, j) - bp_weight_gradients[l](i, j));
                    if (weight_diff > max_diff) {
                        max_diff = weight_diff;
                    }
                }
            }
        }

        // Optionally log the maximum difference
        // std::cout << "Iteration " << iter << ", Max gradient difference: " << max_diff << std::endl;

        // Update parameters using backpropagation gradients
        for (size_t l = 0; l < layers.size(); ++l) {
            layers[l].update_parameters(bp_weight_gradients[l], bp_bias_gradients[l], learning_rate);
        }

        // Compute the cost after the update
        double current_cost = cost(input, target_output);

        // Document the progress if required
        if (convergence_file.is_open()) {
            convergence_file << iter << " " << current_cost << std::endl;
        }

        // Check for convergence
        if (current_cost <= tol_training) {
            break;
        }
    }

    if (convergence_file.is_open()) {
        convergence_file.close();
    }
}

void NeuralNetwork::compute_gradients_finite_difference(const DoubleVector& input, const DoubleVector& target_output,
                                                        std::vector<DoubleMatrix>& weight_gradients, std::vector<DoubleVector>& bias_gradients) {
    const double epsilon = 1e-5;

    // Initialize gradients
    weight_gradients.clear();
    bias_gradients.clear();

    for (size_t l = 0; l < layers.size(); ++l) {
        int output_dim = layers[l].get_output_dim();
        int input_dim = layers[l].get_input_dim();

        DoubleMatrix w_grad(output_dim, input_dim);
        DoubleVector b_grad(output_dim);

        // Compute weight gradients
        for (int i = 0; i < output_dim; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                double original_weight = layers[l].get_weights()(i, j);

                // Perturb weight
                const_cast<DoubleMatrix&>(layers[l].get_weights())(i, j) = original_weight + epsilon;
                double cost_plus = cost(input, target_output);

                const_cast<DoubleMatrix&>(layers[l].get_weights())(i, j) = original_weight - epsilon;
                double cost_minus = cost(input, target_output);

                // Compute gradient
                w_grad(i, j) = (cost_plus - cost_minus) / (2 * epsilon);

                // Restore original weight
                const_cast<DoubleMatrix&>(layers[l].get_weights())(i, j) = original_weight;
            }
        }

        // Compute bias gradients
        for (int i = 0; i < output_dim; ++i) {
            double original_bias = layers[l].get_bias()[i];

            // Perturb bias
            const_cast<DoubleVector&>(layers[l].get_bias())[i] = original_bias + epsilon;
            double cost_plus = cost(input, target_output);

            const_cast<DoubleVector&>(layers[l].get_bias())[i] = original_bias - epsilon;
            double cost_minus = cost(input, target_output);

            // Compute gradient
            b_grad[i] = (cost_plus - cost_minus) / (2 * epsilon);

            // Restore original bias
            const_cast<DoubleVector&>(layers[l].get_bias())[i] = original_bias;
        }

        weight_gradients.push_back(w_grad);
        bias_gradients.push_back(b_grad);
    }
}

void NeuralNetwork::backpropagate(const DoubleVector& input, const DoubleVector& target_output,
                                  std::vector<DoubleMatrix>& weight_gradients, std::vector<DoubleVector>& bias_gradients) {
    // Initialize gradients
    weight_gradients.resize(layers.size());
    bias_gradients.resize(layers.size());

    for (size_t l = 0; l < layers.size(); ++l) {
        weight_gradients[l] = DoubleMatrix(layers[l].get_output_dim(), layers[l].get_input_dim());
        bias_gradients[l] = DoubleVector(layers[l].get_output_dim());
    }

    // Forward pass
    std::vector<DoubleVector> activations;
    std::vector<DoubleVector> zs; // Weighted inputs

    DoubleVector activation = input;
    activations.push_back(activation);

    for (size_t l = 0; l < layers.size(); ++l) {
        DoubleVector z(layers[l].get_output_dim());
        DoubleVector a(layers[l].get_output_dim());

        for (unsigned i = 0; i < layers[l].get_output_dim(); ++i) {
            double sum = layers[l].get_bias()[i];
            for (unsigned j = 0; j < layers[l].get_input_dim(); ++j) {
                sum += layers[l].get_weights()(i, j) * activation[j];
            }
            z[i] = sum;
            a[i] = std::tanh(sum);
        }
        zs.push_back(z);
        activations.push_back(a);
        activation = a;
    }

    // Backward pass
    int num_layers = layers.size();
    DoubleVector delta = activations.back();
    for (unsigned i = 0; i < delta.n(); ++i) {
        delta[i] = (delta[i] - target_output[i]) * (1 - std::pow(std::tanh(zs.back()[i]), 2));
    }

    // Gradient for the last layer
    weight_gradients[num_layers - 1] = DoubleMatrix(delta.n(), activations[num_layers - 1].n());
    bias_gradients[num_layers - 1] = delta;

    for (unsigned i = 0; i < delta.n(); ++i) {
        for (unsigned j = 0; j < activations[num_layers - 1].n(); ++j) {
            weight_gradients[num_layers - 1](i, j) = delta[i] * activations[num_layers - 1][j];
        }
    }

    // Backpropagate the error
    for (int l = num_layers - 2; l >= 0; --l) {
        DoubleVector z = zs[l];
        DoubleVector sp(z.n());
        for (unsigned i = 0; i < z.n(); ++i) {
            sp[i] = 1 - std::pow(std::tanh(z[i]), 2); // Derivative of tanh
        }

        DoubleVector delta_new(layers[l].get_output_dim());
        for (unsigned i = 0; i < delta_new.n(); ++i) {
            double sum = 0.0;
            for (unsigned j = 0; j < delta.n(); ++j) {
                sum += layers[l + 1].get_weights()(j, i) * delta[j];
            }
            delta_new[i] = sum * sp[i];
        }
        delta = delta_new;

        weight_gradients[l] = DoubleMatrix(delta.n(), activations[l].n());
        bias_gradients[l] = delta;

        for (unsigned i = 0; i < delta.n(); ++i) {
            for (unsigned j = 0; j < activations[l].n(); ++j) {
                weight_gradients[l](i, j) = delta[i] * activations[l][j];
            }
        }
    }
}

#endif // PROJECT2_A_H