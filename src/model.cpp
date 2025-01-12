#include <iostream>
#include <filesystem>
#include <numeric>
#include <thread>
#include "model.h"

NeuralHMM::NeuralHMM(int num_states, int num_variants, int input_dimension, int hidden_dimension, const std::string& checkpoint_dir)
    : num_states(num_states),
      num_variants(num_variants),
      input_dimension(input_dimension),
      hidden_dimension(hidden_dimension),
      checkpoint_dir(checkpoint_dir),
      generator(random_device()),
      learning_rate(0.01),
      momentum(0.9)
{
    std::filesystem::create_directories(checkpoint_dir);

    transition_matrix = initialize_transition_matrix();
    emission_matrix = initialize_emission_matrix();
    initialize_weights();
}

double NeuralHMM::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralHMM::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

std::vector<std::vector<double>> NeuralHMM::initialize_transition_matrix() {
    std::vector<std::vector<double>> matrix(num_states, std::vector<double>(num_states));
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    for (auto& row : matrix) {
        double total = 0.0;

        for (auto& val : row) {
            val = distribution(generator);
            total += val;
        }

        for (auto& val : row) {
            val /= total;
        }
    }

    return matrix;
}

std::vector<std::vector<double>> NeuralHMM::initialize_emission_matrix() {
    std::vector<std::vector<double>> matrix(num_states, std::vector<double>(num_variants));
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    for (auto& row : matrix) {
        double total = 0.0;

        for (auto& val : row) {
            val = distribution(generator);
            total += val;
        }

        for (auto& val : row) {
            val /= total;
        }
    }

    return matrix;
}

void NeuralHMM::initialize_weights() {
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    W1.resize(hidden_dimension, std::vector<double>(input_dimension));

    for (auto& row : W1) {
        for (auto& val : row) {
            val = distribution(generator) * std::sqrt(2.0 / (hidden_dimension + input_dimension));
        }
    }

    b1.resize(hidden_dimension, 0.0);
    W2.resize(num_variants, std::vector<double>(hidden_dimension));

    for (auto& row : W2) {
        for (auto& val : row) {
            val = distribution(generator) * std::sqrt(2.0 / (num_variants + hidden_dimension));
        }
    }

    b2.resize(num_variants, 0.0);

    velocity_W1 = std::vector<std::vector<double>>(hidden_dimension, std::vector<double>(input_dimension, 0.0));
    velocity_b1 = std::vector<double>(hidden_dimension, 0.0);
    velocity_W2 = std::vector<std::vector<double>>(num_variants, std::vector<double>(hidden_dimension, 0.0));
    velocity_b2 = std::vector<double>(num_variants, 0.0);
}

double NeuralHMM::normalize_value(const nlohmann::json& value) {
    if (value.is_number_integer() || value.is_number_float()) {
        return value.get<double>();
    } else if (value.is_string()) {
        std::hash<std::string> hasher;
        return static_cast<double>(hasher(value.get<std::string>())) / static_cast<double>(std::numeric_limits<size_t>::max());
    }

    return 0.0;
}

std::vector<std::pair<std::vector<double>, std::vector<double>>> NeuralHMM::load_dataset(const nlohmann::json& data) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> processed;

    for (const auto& item : data) {
        nlohmann::json past_event;

        if (item.contains("data") && item["data"].size() > 1) {
            past_event = item["data"][1];
        }

        nlohmann::json predicted_variants = item.contains("predictions") ? item["predictions"] : nlohmann::json::array();
        auto IO_pair = std::make_pair(prepare_IO(past_event, predicted_variants), std::vector<double>(num_variants, 0.0));

        processed.push_back(IO_pair);
    }

    return processed;
}

std::vector<double> NeuralHMM::prepare_IO(const nlohmann::json& past_event, const nlohmann::json& predicted_variants) {
    std::vector<double> input_data(input_dimension, 0.0);

    if (!past_event.is_null()) {
        int i = 0;

        for (auto iteration = past_event.begin(); iteration != past_event.end() && i < input_dimension / 2; ++iteration, ++i) {
            input_data[i] = normalize_value(iteration.value());
        }
    }

    for (size_t i = 0; i < predicted_variants.size() && i < static_cast<size_t>(input_dimension / 2); ++i) {
        const auto& variant = predicted_variants[i];
        double probability = variant.contains("probability") ? variant["probability"].get<double>() : 0.0;
        input_data[input_dimension / 2 + i] = probability;
    }

    return input_data;
}

std::vector<double> NeuralHMM::forward_propagation(const std::vector<double>& data) {
    std::vector<double> hidden_layer_input(hidden_dimension, 0.0);

    for (int i = 0; i < hidden_dimension; ++i) {
        for (size_t j = 0; j < data.size(); ++j) {
            hidden_layer_input[i] += W1[i][j] * data[j];
        }

        hidden_layer_input[i] += b1[i];
    }

    std::vector<double> hidden_layer(hidden_dimension);

    for (int i = 0; i < hidden_dimension; ++i) {
        hidden_layer[i] = sigmoid(hidden_layer_input[i]);
    }

    std::vector<double> output_layer_input(num_variants, 0.0);

    for (int i = 0; i < num_variants; ++i) {
        for (int j = 0; j < hidden_dimension; ++j) {
            output_layer_input[i] += W2[i][j] * hidden_layer[j];
        }

        output_layer_input[i] += b2[i];
    }

    std::vector<double> output_layer(num_variants);

    for (int i = 0; i < num_variants; ++i) {
        output_layer[i] = sigmoid(output_layer_input[i]);
    }

    return output_layer;
}

void NeuralHMM::train(const nlohmann::json& data, int epochs, int batch_size, double wait_time) {
    auto training_data = load_dataset(data);

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;

    for (const auto& item : training_data) {
        X.push_back(item.first);
        y.push_back(item.second);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < X.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, X.size());

            std::vector<std::vector<double>> X_batch(X.begin() + i, X.begin() + end);
            std::vector<std::vector<double>> y_batch(y.begin() + i, y.begin() + end);

            std::vector<std::vector<double>> predictions;

            for (const auto& x : X_batch) {
                predictions.push_back(forward_propagation(x));
            }

            double batch_loss = 0.0;

            for (size_t j = 0; j < predictions.size(); ++j) {
                double loss = 0.0;

                for (int k = 0; k < num_variants; ++k) {
                    double prediction = predictions[j][k];
                    double target = y_batch[j][k];

                    loss -= target * std::log(prediction + 1e-8) + (1 - target) * std::log(1 - prediction + 1e-8);
                }

                batch_loss += loss;
            }

            total_loss += batch_loss / predictions.size();

            backpropagate(X_batch, predictions, y_batch);

            if (wait_time > 0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
            }
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch: " << epoch << "/" << epochs << ", Average Loss: " << total_loss / X.size() << std::endl;
        }
    }
}

void NeuralHMM::backpropagate(const std::vector<std::vector<double>>& X_batch, const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) {
    for (size_t batch_index = 0; batch_index < X_batch.size(); ++batch_index) {
        const auto& x = X_batch[batch_index];
        const auto& prediction = predictions[batch_index];
        const auto& target = targets[batch_index];

        std::vector<double> output_error(prediction.size());

        for (size_t i = 0; i < prediction.size(); ++i) {
            output_error[i] = prediction[i] - target[i];
        }

        std::vector<double> output_delta(prediction.size());

        for (size_t i = 0; i < output_delta.size(); ++i) {
            output_delta[i] = output_error[i] * sigmoid_derivative(prediction[i]);
        }

        std::vector<double> hidden_layer_input(hidden_dimension, 0.0);

        for (int i = 0; i < hidden_dimension; ++i) {
            for (int j = 0; j < input_dimension; ++j) {
                hidden_layer_input[i] += W1[i][j] * x[j];
            }

            hidden_layer_input[i] += b1[i];
        }

        std::vector<double> hidden_layer(hidden_dimension);

        for (int i = 0; i < hidden_dimension; ++i) {
            hidden_layer[i] = sigmoid(hidden_layer_input[i]);
        }

        std::vector<double> hidden_error(hidden_dimension, 0.0);

        for (int i = 0; i < hidden_dimension; ++i) {
            for (int j = 0; j < num_variants; ++j) {
                hidden_error[i] += W2[j][i] * output_delta[j];
            }
        }

        std::vector<double> hidden_delta(hidden_dimension);

        for (int i = 0; i < hidden_dimension; ++i) {
            hidden_delta[i] = hidden_error[i] * sigmoid_derivative(hidden_layer[i]);
        }

        std::vector<std::vector<double>> dW2(num_variants, std::vector<double>(hidden_dimension, 0.0));

        for (int i = 0; i < num_variants; ++i) {
            for (int j = 0; j < hidden_dimension; ++j) {
                dW2[i][j] = output_delta[i] * hidden_layer[j];
            }
        }

        std::vector<double> db2 = output_delta;
        std::vector<std::vector<double>> dW1(hidden_dimension, std::vector<double>(input_dimension, 0.0));

        for (int i = 0; i < hidden_dimension; ++i) {
            for (int j = 0; j < input_dimension; ++j) {
                dW1[i][j] = hidden_delta[i] * x[j];
            }
        }

        std::vector<double> db1 = hidden_delta;

        for (int i = 0; i < num_variants; ++i) {
            for (int j = 0; j < hidden_dimension; ++j) {
                velocity_W2[i][j] = momentum * velocity_W2[i][j] - learning_rate * dW2[i][j];
            }
        }

        for (int i = 0; i < num_variants; ++i) {
            velocity_b2[i] = momentum * velocity_b2[i] - learning_rate * db2[i];
        }

        for (int i = 0; i < hidden_dimension; ++i) {
            for (int j = 0; j < input_dimension; ++j) {
                velocity_W1[i][j] = momentum * velocity_W1[i][j] - learning_rate * dW1[i][j];
            }
        }

        for (int i = 0; i < hidden_dimension; ++i) {
            velocity_b1[i] = momentum * velocity_b1[i] - learning_rate * db1[i];
        }

        // Updates
        for (int i = 0; i < num_variants; ++i) {
            for (int j = 0; j < hidden_dimension; ++j) {
                W2[i][j] += velocity_W2[i][j];
            }
        }

        for (int i = 0; i < num_variants; ++i) {
            b2[i] += velocity_b2[i];
        }

        for (int i = 0; i < hidden_dimension; ++i) {
            for (int j = 0; j < input_dimension; ++j) {
                W1[i][j] += velocity_W1[i][j];
            }
        }

        for (int i = 0; i < hidden_dimension; ++i) {
            b1[i] += velocity_b1[i];
        }
    }
}

void NeuralHMM::save_checkpoint(int epoch, double loss) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);

    std::stringstream filename;
    filename << checkpoint_dir << "/epoch_" << epoch << "_loss_" << std::fixed << std::setprecision(4) << loss << "_" << std::put_time(std::localtime(&timestamp), "%Y%m%d_%H%M%S") << ".cp";

    std::ofstream checkpoint(filename.str(), std::ios::binary);

    if (checkpoint) {
        checkpoint.write(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(double));
        checkpoint.write(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(double));
        checkpoint.write(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(double));
        checkpoint.write(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(double));

        std::cout << "Checkpoint saved: " << filename.str() << std::endl;
    } else {
        std::cerr << "Could not open checkpoint file for writing: " << filename.str() << std::endl;
    }
}

void NeuralHMM::load_checkpoint(const std::string& filename) {
    std::ifstream checkpoint(filename, std::ios::binary);

    if (!checkpoint) {
        std::cerr << "Could not open checkpoint file for reading: " << filename << std::endl;
        return;
    }

    try {
        checkpoint.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(double));
        checkpoint.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(double));
        checkpoint.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(double));
        checkpoint.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(double));

        std::cout << "Checkpoint loaded: " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error while loading checkpoint: " << e.what() << std::endl;
    }
}

std::vector<std::pair<int, double>> NeuralHMM::predict(const nlohmann::json& past_event, int num_predictions) {
    std::vector<double> input_data = prepare_IO(past_event, nlohmann::json::array());
    std::vector<double> predictions = forward_propagation(input_data);
    std::vector<std::pair<int, double>> variant_predictions;

    for (size_t i = 0; i < predictions.size(); ++i) {
        variant_predictions.emplace_back(i, predictions[i]);
    }

    std::sort(variant_predictions.begin(), variant_predictions.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return a.second > b.second;
    });

    if (variant_predictions.size() > num_predictions) {
        variant_predictions.resize(num_predictions);
    }

    return variant_predictions;
}

std::vector<std::pair<int, double>> NeuralHMM::predict_from_prompt(const std::string& prompt, int num_predictions) {
    try {
        auto data = nlohmann::json::parse(prompt);

        auto context = data.value("context", nlohmann::json{});
        auto features = data.value("features", nlohmann::json{});

        std::vector<double> input_data = prepare_IO(context, features);
        std::vector<double> predictions = forward_propagation(input_data);
        std::vector<std::pair<int, double>> variant_predictions;

        for (size_t i = 0; i < predictions.size(); ++i) {
            std::string description = features.contains("variant_" + std::to_string(i)) ? features["variant_" + std::to_string(i)].get<std::string>() : "Variant " + std::to_string(i);
            variant_predictions.emplace_back(i, predictions[i]);
        }

        std::sort(variant_predictions.begin(), variant_predictions.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second > b.second;
        });

        if (variant_predictions.size() > num_predictions) {
            variant_predictions.resize(num_predictions);
        }

        return variant_predictions;
    } catch (const nlohmann::json::parse_error& e) {
        std::cout << "Invalid prompt format. JSON required." << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cout << "Error while prediction from prompt: " << e.what() << std::endl;
        return {};
    }
}