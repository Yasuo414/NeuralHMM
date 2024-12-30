import numpy
import json
import os
import datetime
import pickle
import time

class HMM:
    def __init__(self, num_states, num_variants, input_dimension, hidden_dimension, checkpoint_dir):
        self.num_states = num_states
        self.num_variants = num_variants
        
        self.transition_matrix = self._initialize_transition_matrix()
        self.emission_matrix = self._initialize_emission_matrix()

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._initialize_weights()

    def _initialize_weights(self):
        self.W1 = numpy.random.randn(self.hidden_dimension, self.input_dimension) * numpy.sqrt(2.0 / (self.input_dimension + self.hidden_dimension))
        self.b1 = numpy.zeros(self.hidden_dimension)
        self.W2 = numpy.random.randn(self.num_variants, self.hidden_dimension) * numpy.sqrt(2.0 / (self.hidden_dimension + self.num_variants))
        self.b2 = numpy.zeros(self.num_variants)
        
        self.learning_rate = 0.01
        self.momentum = 0.9
        
        self.velocity_W1 = numpy.zeros_like(self.W1)
        self.velocity_b1 = numpy.zeros_like(self.b1)
        self.velocity_W2 = numpy.zeros_like(self.W2)
        self.velocity_b2 = numpy.zeros_like(self.b2)

    def _initialize_transition_matrix(self):
        matrix = numpy.random.rand(self.num_states, self.num_states)

        return matrix / matrix.sum(axis=1, keepdims=True)
    
    def _initialize_emission_matrix(self):
        matrix = numpy.random.rand(self.num_states, self.num_variants)

        return matrix / matrix.sum(axis=1, keepdims=True)
    
    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def load_dataset(self, data):
        if isinstance(data, str):
            training_data = json.loads(data)
        else:
            training_data = data
        
        processed_data = []

        for item in training_data:
            past_event = item["data"][1] if len(item["data"]) > 1 else None
            predicted_variants = item.get("predictions", [])

            input_data, target_variants = self._prepare_IO(past_event, predicted_variants)

            processed_data.append({
                "input": input_data,
                "target": target_variants
            })
        
        return processed_data
    
    def _prepare_IO(self, past_event, predicted_variants):
        input_data = numpy.zeros(self.input_dimension)

        if past_event:
            for i, (key, value) in enumerate(past_event.items()):
                if i < self.input_dimension // 2:
                    input_data[i] = float(value) if isinstance(value, (int, float)) else hash(str(value)) % 1000 / 1000
        
        target_variants = numpy.zeros(self.num_variants)
        for i, variant in enumerate(predicted_variants):
            if i < self.num_variants:
                target_variants[i] = variant.get("probability", 0)

                if i < self.input_dimension // 2:
                    input_data[self.input_dimension // 2 + i] = variant.get("probability", 0)
        
        return input_data, target_variants
    
    def forward_propagation(self, input_data):
        hidden_layer = self.sigmoid(numpy.dot(self.W1, input_data) + self.b1)
        output_layer = self.sigmoid(numpy.dot(self.W2, hidden_layer) + self.b2)

        return output_layer
    
    def train(self, data, epochs=100, batch_size=32, wait_time=0):
        training_data = self.load_dataset(data)

        X = numpy.array([item["input"] for item in training_data])
        y = numpy.array([item["target"] for item in training_data])

        for epoch in range(epochs):
            total_loss = 0

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                predictions = numpy.array([self.forward_propagation(x) for x in X_batch])
                loss = numpy.mean(-numpy.sum(y_batch * numpy.log(predictions + 1e-8) + (1 - y_batch) * numpy.log(1 - predictions + 1e-8), axis=1))

                total_loss += loss

                self._backpropagate(X_batch, predictions, y_batch)
                
                if wait_time > 0:
                    time.sleep(wait_time)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{epochs}, Average Loss: {total_loss / len(X)}")
            
            if epoch % 100000 == 0:
                print("Transition Matrix:")
                print(self.transition_matrix)
                print("\nEmission Matrix:")
                print(self.emission_matrix)
                print("\nLayer 1 Weights:")
                print("W1:\n", self.W1)
                print("b1:\n", self.b1)
                print("\nLayer 2 Weights:")
                print("W2:\n", self.W2)
                print("b2:\n", self.b2)

                self.save_ckpt(epoch, loss)
    
    def _backpropagate(self, X_batch, predictions, targets):
        for x, prediction, target in zip(X_batch, predictions, targets):
            output_error = prediction - target
            output_delta = output_error * self.sigmoid_derivative(prediction)

            hidden_layer = self.sigmoid(numpy.dot(self.W1, x) + self.b1)
            hidden_error = numpy.dot(self.W2.T, output_delta)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer)

            dW2 = numpy.outer(output_delta, hidden_layer)
            db2 = output_delta
            dW1 = numpy.outer(hidden_delta, x)
            db1 = hidden_delta

            self.velocity_W2 = self.momentum * self.velocity_W2 - self.learning_rate * dW2
            self.velocity_b2 = self.momentum * self.velocity_b2 - self.learning_rate * db2
            self.velocity_W1 = self.momentum * self.velocity_W1 - self.learning_rate * dW1
            self.velocity_b1 = self.momentum * self.velocity_b1 - self.learning_rate * db1

            self.W2 += self.velocity_W2
            self.b2 += self.velocity_b2
            self.W1 += self.velocity_W1
            self.b1 += self.velocity_b1
    
    def predict(self, past_event, num_predictions=5):
        input_data, _ = self._prepare_IO(past_event, [])
        predictions = self.forward_propagation(input_data)

        variant_predictions = [
            {
                "index": i,
                "probability": float(prediction),
                "description": f"Varianta {i}"
            }
            for i, prediction in enumerate(predictions)
        ]

        variant_predictions.sort(key=lambda x: x["probability"], reverse=True)

        return variant_predictions[:num_predictions]
    
    def save_ckpt(self, epoch, loss):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        state_dict = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "epoch": epoch,
            "loss": loss
        }

        with open(f"checkpoints\\epoch_{epoch}_loss_{loss:.4f}_{timestamp}.cp", "wb") as f:
            pickle.dump(state_dict, f)

        print(f"Checkpoint saved: epoch_{epoch}_loss_{loss:.4f}_{timestamp}.cp")
    
    def load_ckpt(self, filename=None):
        if filename is None:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]

            if not checkpoints:
                print("Checkpoints not found.")
                return False
            
            filename = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)))
        
        filepath = os.path.join(self.checkpoint_dir, filename)

        try:
            checkpoint = numpy.load(filepath)

            self.W1 = checkpoint["W1"]
            self.b1 = checkpoint["b1"]
            self.W2 = checkpoint["W2"]
            self.b2 = checkpoint["b2"]

            print(f"Checkpoint loaded: {filename}")
            print(f"Epoch: {checkpoint["epoch"]}, Loss: {checkpoint["loss"]:.4f}")

            return True
        
        except Exception as e:
            print(f"Error while loading checkpoint: {e}")
            return False
    
    def predict_from_prompt(self, prompt, num_predictions=5):
        try:
            prompt_data = json.loads(prompt)

            context = prompt_data.get("context", {})
            features = prompt_data.get("features", {})

            input_data = self._prepare_prompt_input(context, features)

            predictions = self.forward_propagation(input_data)

            variant_predictions = [
                {
                    "index": i,
                    "probability": float(prediction),
                    "description": features.get(f"variant_{i}", f"Variant {i}")
                }
                for i, prediction in enumerate(predictions)
            ]

            variant_predictions.sort(key=lambda x: x["probability"], reverse=True)

            return variant_predictions[:num_predictions]
        except json.JSONDecodeError:
            print("Invalid prompt format. JSON required.")
            return []
        except Exception as e:
            print(f"Error while prediction from prompt: {e}")
            return []
    
    def _prepare_prompt_input(self, context, features):
        input_data = numpy.zeros(self.input_dimension)

        for i, (key, value) in enumerate(context.items()):
            if i < self.input_dimension // 2:
                input_data[i] = self._normalize_value(value)
        
        feature_keys = list(features.keys())
        for i, key in enumerate(feature_keys):
            if i < self.input_dimension // 2:
                input_data[self.input_dimension // 2 + i] = self._normalize_value(features[key])
        
        return input_data
    
    def _normalize_value(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return hash(value) % 1000 / 1000
        else:
            return 0.0

if __name__ == "__main__":
    num_states = 5
    num_variants = 10
    input_dim = 20
    hidden_dim = 10

    model = HMM(num_states, num_variants, input_dim, hidden_dim, checkpoint_dir="checkpoints")

    # Training dataset
    training_json = json.dumps([
        {
            'data': [
                None,
                {
                    'location': 'Prague',
                    'temperature': 25,
                    'weather': 'sunny'
                }
            ],
            'predictions': [
                {'probability': 0.7, 'description': 'Temperature rises'},
                {'probability': 0.3, 'description': 'Temperature drops'}
            ]
        }
    ])
    model.train(training_json, epochs=20109910, batch_size=32, wait_time=0)

    # Saving a checkpoint
    checkpoint_file = model.save_ckpt(epoch=100000, loss=0.1)

    # Example of a prompt for prediction
    prompt = json.dumps({
        'context': {
            'location': 'Prague',
            'season': 'summer'
        },
        'features': {
            'variant_0': 'High Temperatures',
            'variant_1': 'Raining',
            'variant_2': 'Storms',
            'variant_3': 'Colder'
        }
    })

    # Load checkpoint and prediction
    model.load_ckpt(checkpoint_file)
    predictions = model.predict_from_prompt(prompt)
    
    print("Predicted variants:")
    for pred in predictions:
        print(f"Index: {pred['index']}, Probability: {pred['probability']:.2f}, Description: {pred['description']}")