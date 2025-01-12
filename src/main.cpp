#include <iostream>
#include <string>
#include "model.h"
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    int num_states = 5;
    int num_variants = 10;
    int input_dimension = 20;
    int hidden_dimension = 10;

    NeuralHMM model(num_states, num_variants, input_dimension, hidden_dimension, "checkpoints");

    bool train_mode = false;
    std::string path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--train") {
            train_mode = true;

            if (i + 1 < argc) {
                path = argv[i + 1];
                i++;
            } else {
                std::cerr << "Error: file path is expected after the '--train' argument." << std::endl;
                return 1;
            }
        }
    }

    if (train_mode) {
        nlohmann::json training_data = nlohmann::json::array();
        training_data.push_back({
            {
                "data", {
                    nullptr,
                    {
                        {
                            "location",
                            "Prague"
                        },
                        {
                            "temperature",
                            25
                        },
                        {
                            "weather",
                            "sunny"
                        }
                    }
               }
           },
        {
            "predictions", {
            {
                {
                    "probability",
                    0.7
                },
                {
                    "description",
                    "Temperature rises"
                }
            },
            {
                {
                    "probability",
                    0.3
                },
                {
                    "description",
                    "Temperature drops"
                }
            }
        }}
    });

    model.train(training_data.dump(), 20109910, 32, 0);
    model.save_checkpoint(100000, 0.1);
    }
}