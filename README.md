# NeuralHMM Documentation

## Overview

This repository contains the source code for a context-dependent Hidden Markov Model (HMM) enhanced with AI capabilities. The model predicts future events based on past occurrences and can optionally refute conjectures. 

While the primary use case is weather prediction, you can adapt the model to other scenarios. If you use it, **please leave the credit intact**:  
Created by [onnx4144](https://www.instagram.com/onnx4144/).  

Thank you in advance for your support!

---

## Included Files

### `neural_hmm.py`
This file contains the entire HMM model implemented from scratch using the NumPy library and other Python built-ins. It is written in **Python 3.12** and provides a robust implementation for your use.

### `pytorch_hmm.py`
This is a modified version of the model, fully compatible with the PyTorch library. However, this version is less stable due to checkpointing issues during inference. While training is successful, inference may throw errors, such as:

C:\Users%username%\HMMPrediction\pytorch_hmm.py:278: FutureWarning: You are using torch.load with weights_only=False... AttributeError: 'collections.OrderedDict' object has no attribute 'predict_from_prompt'


For this reason, I **recommend using `neural_hmm.py`** unless you specifically need PyTorch compatibility.

---

## Usage Instructions

### Running the Model

To run the model, use the following commands in your terminal:

- For the NumPy version:  
  ```bash
  python neural_hmm.py
For the PyTorch version:
python pytorch_hmm.py
Required Libraries
Install the following Python libraries using pip:

NumPy
Matplotlib
For the PyTorch version, visit the PyTorch website and follow these steps:

Select the stable build.
Choose your operating system.
Choose the Pip package.
Select the appropriate Python version.
For NVIDIA GPUs, choose the latest CUDA version (e.g., CUDA 12.4). For CPU-only systems, select the CPU option.
Copy the generated pip3 installation command and run it in your terminal.

Training Details

Training the Model
During training, every ~10,000 epochs, a visualization window will appear. This window displays the model's progress using colored squares. The window remains open for 5 seconds before disappearing.

You can customize the training parameters:

Epochs: Feel free to increase beyond 10,000. For example, 29,000,000 epochs with a batch size of 32 is feasible.
Performance: 10,000 epochs are completed in approximately 1 minute.
Checkpoint Size: Checkpoints are very small (around 1036 bytes).
Recommendations
Use the neural_hmm.py script for reliable results.
The PyTorch version is experimental and may require modifications for stable inference.
Future Updates

I plan to release future versions of the model with the following improvements:

A fixed checkpointing system.
Inference via a separate script.
Additional adjustable parameters.
Feedback

I would greatly appreciate your feedback and ideas for improvement. While it's not mandatory, any suggestions will help make the model better.

Thank you for your time, and enjoy working with NeuralHMM!
