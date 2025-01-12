#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <nlohmann/json.hpp>

class NeuralHMM {
private:
    int num_states;
    int num_variants;
    int input_dimension;
    int hidden_dimension;
    std::string checkpoint_dir;

    std::vector<std::vector<double>> transition_matrix;
    std::vector<std::vector<double>> emission_matrix;
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b2;

    double learning_rate;
    double momentum;
    std::vector<std::vector<double>> velocity_W1;
    std::vector<double> velocity_b1;
    std::vector<std::vector<double>> velocity_W2;
    std::vector<double> velocity_b2;

    std::random_device random_device;
    std::mt19937 generator;

    void initialize_weights();
    std::vector<std::vector<double>> initialize_transition_matrix();
    std::vector<std::vector<double>> initialize_emission_matrix();

    double sigmoid(double x);
    double sigmoid_derivative(double x);
    std::vector<double> prepare_IO(const nlohmann::json& past_event, const nlohmann::json& predicted_variants);
    std::vector<double> prepare_prompt_input(const nlohmann::json& context, const nlohmann::json& features);
    double normalize_value(const nlohmann::json& value);
    void backpropagate(const std::vector<std::vector<double>>& X_batch, const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

public:
    NeuralHMM(int num_states, int num_variants, int input_dimension, int hidden_dimension, const std::string& checkpoint_dir);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> load_dataset(const nlohmann::json& data);
    void train(const nlohmann::json& data, int epochs = 100, int batch_size = 32, double wait_time = 0);

    std::vector<std::pair<int, double>> predict(const nlohmann::json& past_event, int num_predictions = 5);
    std::vector<std::pair<int, double>> predict_from_prompt(const std::string& prompt, int num_predictions = 5);

    void save_checkpoint(int epoch, double loss);
    void load_checkpoint(const std::string& filename);

private:
    std::vector<double> forward_propagation(const std::vector<double>& data);
};

#endif