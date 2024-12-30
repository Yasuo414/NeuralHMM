import matplotlib.pyplot
import torch
import json
import os
import datetime
import numpy as np
import glob
import matplotlib

class HMM(torch.nn.Module):
    def __init__(self, num_states, num_variants, input_dimension, hidden_dimension, checkpoint_dir):
        super().__init__()
        self.num_states = num_states
        self.num_variants = num_variants
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'ckpt'), exist_ok=True)
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.Sigmoid()
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, num_variants),
            torch.nn.Sigmoid()
        )
        
        self._initialize_weights()
        
        self.learning_rate = 0.01
        self.momentum = 0.9
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum
        )
        
        self.loss_fn = torch.nn.BCELoss()
    
    def _initialize_weights(self):
        for layer in [self.layer1, self.layer2]:
            for module in layer:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight)
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
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
                "input": torch.FloatTensor(input_data),
                "target": torch.FloatTensor(target_variants)
            })
        
        return processed_data
    
    def _prepare_IO(self, past_event, predicted_variants):
        input_data = np.zeros(self.input_dimension)
        
        if past_event:
            for i, (key, value) in enumerate(past_event.items()):
                if i < self.input_dimension // 2:
                    input_data[i] = float(value) if isinstance(value, (int, float)) else hash(str(value)) % 1000 / 1000
        
        target_variants = np.zeros(self.num_variants)
        for i, variant in enumerate(predicted_variants):
            if i < self.num_variants:
                target_variants[i] = variant.get("probability", 0)
                
                if i < self.input_dimension // 2:
                    input_data[self.input_dimension // 2 + i] = variant.get("probability", 0)
        
        return input_data, target_variants
    
    def train(self, data, epochs=100, batch_size=32, wait_time=0):
        training_data = self.load_dataset(data)
        indices = torch.randperm(len(training_data))
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i in range(0, len(training_data), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = torch.stack([training_data[j]["input"] for j in batch_indices])
                y_batch = torch.stack([training_data[j]["target"] for j in batch_indices])
                
                self.optimizer.zero_grad()
                
                outputs = self(X_batch)
                
                loss = self.loss_fn(outputs, y_batch)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10000 == 0:
                print(f"Epoch: {epoch}/{epochs}, Average Loss: {total_loss / len(training_data)}")
            
            if epoch % 100_000 == 0:
                self._visualize_weights(epoch)
                self._save_checkpoint(epoch, total_loss)
    
    def _save_checkpoint(self, epoch, loss):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ckpt/checkpoint_epoch_{epoch}_loss_{loss:.4f}_{timestamp}.ckpt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {filename}")
    
    def merge_checkpoints(self, output_path='final_model.pth'):
        checkpoint_files = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, 'ckpt', '*.ckpt')), 
            key=os.path.getctime
        )
        
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint)
            
            torch.save(checkpoint['model_state_dict'], 
                       os.path.join(self.checkpoint_dir, output_path))
            
            print(f"Final model saved to {output_path}")
    
    def _visualize_weights(self, epoch):
        matplotlib.pyplot.figure(figsize=(15, 5))

        matplotlib.pyplot.subplot(121)
        matplotlib.pyplot.title(f'Layer 1 Weights (Epoch {epoch})')
        matplotlib.pyplot.imshow(self.layer1[0].weight.detach().numpy(), aspect='auto', cmap='viridis')
        matplotlib.pyplot.colorbar()

        matplotlib.pyplot.subplot(122)
        matplotlib.pyplot.title(f'Layer 2 Weights (Epoch {epoch})')
        matplotlib.pyplot.imshow(self.layer2[0].weight.detach().numpy(), aspect='auto', cmap='viridis')
        matplotlib.pyplot.colorbar()
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.pause(3)
        matplotlib.pyplot.close()
    
    def predict(self, past_event, num_predictions=5):
        input_data, _ = self._prepare_IO(past_event, [])
        input_tensor = torch.FloatTensor(input_data)
        
        with torch.no_grad():
            predictions = self(input_tensor)
        
        variant_predictions = [
            {
                "index": i,
                "probability": float(prediction),
                "description": f"Varianta {i}"
            }
            for i, prediction in enumerate(predictions)
        ]
        
        return sorted(variant_predictions, key=lambda x: x["probability"], reverse=True)[:num_predictions]
    
    def predict_from_prompt(self, prompt, num_predictions=5):
        try:
            prompt_data = json.loads(prompt)
            
            context = prompt_data.get("context", {})
            features = prompt_data.get("features", {})
            
            input_data = self._prepare_prompt_input(context, features)
            input_tensor = torch.FloatTensor(input_data)
            
            with torch.no_grad():
                predictions = self(input_tensor)
            
            variant_predictions = [
                {
                    "index": i,
                    "probability": float(prediction),
                    "description": features.get(f"variant_{i}", f"Variant {i}")
                }
                for i, prediction in enumerate(predictions)
            ]
            
            return sorted(variant_predictions, key=lambda x: x["probability"], reverse=True)[:num_predictions]
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def _prepare_prompt_input(self, context, features):
        input_data = np.zeros(self.input_dimension)
        
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

    model = torch.load("checkpoints\\final_model.pth")

    prompt = json.dumps({
        'context': {
            'location': 'Prague',
            'season': 'summer'
        },
        'features': {
            'variant_0': 'High temperatures',
            'variant_1': 'Rain',
            'variant_2': 'Thunderstorms',
            'variant_3': 'Cooling'
        }
    })

    # Make predictions
    predictions = model.predict_from_prompt(prompt)
    
    print("Predicted variants:")
    for pred in predictions:
        print(f"Index: {pred['index']}, Probability: {pred['probability']:.2f}, Description: {pred['description']}")