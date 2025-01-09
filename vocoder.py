import scipy.signal
import librosa
import os
import typing
import pickle
import gzip
import numpy
import wave
import yaml

class Vocoder:
    def __init__(self, mel_channels: int = 80, hidden_channels: int = 512, kernel_size: int = 7, n_layers: int = 16, upsample_scales=[8, 8, 2, 2], target_sample_rate: int = 44100):
        self.mel_channels = mel_channels
        self.hidden_channels = hidden_channels
        self.target_sample_rate = target_sample_rate
        self.upsample_scales = upsample_scales
        self.total_scale = numpy.prod(upsample_scales)

        self.encoder_layers = []
        self.decoder_layers = []

        current_channels = mel_channels

        def xavier_initialization(shape):
            return numpy.random.randn(*shape) * numpy.sqrt(2.0 / shape[0] + shape[1])

        for _ in range(4):
            layer = {
                "conv_down": numpy.random.randn(hidden_channels, current_channels, kernel_size) * 0.02,
                "conv_process": numpy.random.randn(hidden_channels, hidden_channels, kernel_size) * 0.02,
                "pool_factor": 2
            }

            self.encoder_layers.append(layer)
            current_channels = hidden_channels

        self.bottleneck = {
            "conv1": numpy.random.randn(hidden_channels*2, hidden_channels, kernel_size) * 0.02,
            "conv2": numpy.random.randn(hidden_channels*2, hidden_channels*2, kernel_size) * 0.02
        }

        for _ in range(4):
            layer = {
                "conv_up": numpy.random.randn(hidden_channels//2, hidden_channels, kernel_size) * 0.02,
                "conv_process": numpy.random.randn(hidden_channels//2, hidden_channels, kernel_size) * 0.02,
                "upsample_factor": 2
            }

            self.decoder_layers.append(layer)

        self.output_conv = numpy.random.randn(1, hidden_channels//2, 1) * 0.02

        self.disc_layers = []
        disc_channels = [64, 128, 256, 512]

        in_channels = 1
        for channels in disc_channels:
            layer = {
                "conv": numpy.random.randn(channels, in_channels, kernel_size) * 0.02,
                "pool_factor": 2
            }

            self.disc_layers.append(layer)
            in_channels = channels

        self.disc_output = numpy.random.randn(1, disc_channels[-1], 1) * 0.02


        self.anti_anti_aliasing_filter = self._design_anti_aliasing_filter()

        self.history = {
            "gen_losses": [],
            "disc_losses": [],
            "epochs": 0
        }

    def _design_anti_aliasing_filter(self):
        nyquist = self.target_sample_rate / 2
        cutoff = 20000
        transition_width = 2000

        wp = cutoff / nyquist
        ws = (cutoff + transition_width) / nyquist

        N, _ = scipy.signal.kaiserord(80, transition_width / nyquist)
        taps = scipy.signal.firwin(N, cutoff / nyquist, fs=self.target_sample_rate)
        taps = scipy.signal.firwin(N, cutoff, fs=self.target_sample_rate)

        return taps

    def _conv1d(self, x: numpy.ndarray, kernel: numpy.ndarray, dilation: int = 1) -> numpy.ndarray:
        def gelu_activation(x):
            return x * 0.5 * (1 + numpy.tanh(numpy.sqrt(2/numpy.pi) * (x + 0.044715 * x**3)))
        
        batch, channels, length = x.shape
        kernel_size = kernel.shape[-1]

        pad_size = (kernel_size - 1) * dilation
        x_pad = numpy.pad(x, ((0, 0), (0, 0), (pad_size//2, pad_size//2)))

        output = numpy.zeros((batch, kernel.shape[0], length))

        for b in range(batch):
            for o in range(kernel.shape[0]):
                for i in range(channels):
                    dilation_kernel = numpy.zeros(1 + (kernel_size - 1) * dilation)
                    dilation_kernel[::dilation] = kernel[o, i]
                    output[b, o] += scipy.signal.convolve(x_pad[b, i], dilation_kernel, mode="valid")

        return gelu_activation(output)

    def _downsample(self, x: numpy.ndarray, factor: int) -> numpy.ndarray:
        return scipy.signal.decimate(x, factor, axis=-1)

    def _upsample(self, x: numpy.ndarray, factor: int) -> numpy.ndarray:
        return scipy.signal.resample_poly(x, factor, 1, axis=-1)

    def generate(self, mel_spec: numpy.ndarray) -> numpy.ndarray:
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec[numpy.newaxis, ...]

        x = numpy.transpose(mel_spec, (0, 2, 1))

        skip_connections = []
        for layer in self.encoder_layers:
            x = self._conv1d(x, layer["conv_down"])
            x = self._conv1d(x, layer["conv_process"])
            x = numpy.maximum(0.1 * x, x)
            skip_connections.append(x)
            x = self._downsample(x, layer["pool_factor"])

        x = self._conv1d(x, self.bottleneck["conv1"])
        x = numpy.maximum(0.1 * x, x)
        x = self._conv1d(x, self.bottleneck["conv2"])
        x = numpy.maximum(0.1 * x, x)

        for layer, skip in zip(self.decoder_layers, reversed(skip_connections)):
            x = self._upsample(x, layer["upsample_factor"])
            x = self._conv1d(x, layer["conv_up"])
            x = numpy.maximum(0.1 * x, x)

            x = numpy.concatenate([x, skip], axis=1)

            x = self._conv1d(x, layer["conv_process"])
            x = numpy.maximum(0.1 * x, x)

        x = self._conv1d(x, self.output_conv)
        audio = numpy.tanh(x)

        if self.target_sample_rate != 44100:
            audio = librosa.resample(audio, orig_sr=self.target_sample_rate, target_sr=44100)

        return audio.squeeze()

    def discriminate(self, audio: numpy.ndarray) -> numpy.ndarray:
        if len(audio.shape) == 1:
            audio = audio[numpy.newaxis, numpy.newaxis, :]
        elif len(audio.shape) == 2:
            audio = audio[numpy.newaxis, ...]

        x = audio

        for layer in self.disc_layers:
            x = self._conv1d(x, layer["conv"])
            x = numpy.maximum(0.1 * x, x)
            x = self._downsample(x, layer["pool_factor"])

        x = self._conv1d(x, self.disc_output)

        return x.squeeze()

    def _compute_numerical_gradient(self, weight: numpy.ndarray, loss_fn, epsilon=1e-7) -> numpy.ndarray:
        gradient = numpy.zeros_like(weight)
        it = numpy.nditer(weight, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            index = it.multi_index
            old_value = weight[index]

            weight[index] = old_value + epsilon
            loss_plus = loss_fn()

            weight[index] = old_value - epsilon
            loss_minus = loss_fn()

            gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)

            weight[index] = old_value
            it.iternext()

        return gradient

    def _update_weights(self, weights: numpy.ndarray, gradients: numpy.ndarray, learning_rate: float, momentum: float = 0.9) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        velocity = getattr(self, "_velocity", numpy.zeros_like(weights))
        velocity = momentum * velocity - learning_rate * gradients
        weights += velocity

        return weights, velocity

    def save_checkpoint(self, path: str, compress: bool = True) -> None:
        checkpoint = {
            "config": {
                "mel_channels": self.mel_channels,
                "hidden_channels": self.hidden_channels,
                "target_sample_rate": self.target_sample_rate,
                "upsample_scales": self.upsample_scales
            },
            "encoder_layers": self.encoder_layers,
            "bottleneck": self.bottleneck,
            "decoder_layers": self.decoder_layers,
            "output_conv": self.output_conv,
            "disc_layers": self.disc_layers,
            "disc_output": self.disc_output,
            "history": self.history
        }

        if compress:
            with gzip.open(path, "wb") as f:
                pickle.dump(checkpoint, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "Vocoder":
        try:
            with gzip.open(path, "rb") as f:
                checkpoint = pickle.load(f)
        except:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

        instance = cls(**checkpoint["config"])

        instance.encoder_layers = checkpoint["encoder_layers"]
        instance.bottleneck = checkpoint["bottleneck"]
        instance.decoder_layers = checkpoint["decoder_layers"]
        instance.output_conv = checkpoint["output_conv"]
        instance.disc_layers = checkpoint["disc_layers"]
        instance.disc_output = checkpoint["disc_output"]
        instance.history = checkpoint["history"]

        return instance

    def export_model(self, path: str) -> None:
        model = {
            "config": {
                "mel_channels": self.mel_channels,
                "hidden_channels": self.hidden_channels,
                "target_sample_rate": self.target_sample_rate,
                "upsample_scales": self.upsample_scales
            },
            "encoder_layers": [{
                "conv_down": layer["conv_down"].astype(numpy.float16),
                "conv_process": layer["conv_process"].astype(numpy.float16),
                "pool_factor": layer["pool_factor"]
            } for layer in self.encoder_layers],
            "bottleneck": {
                "conv1": self.bottleneck["conv1"].astype(numpy.float16),
                "conv2": self.bottleneck["conv2"].astype(numpy.float16)
            },
            "decoder_layers": [{
                "conv_up": layer["conv_up"].astype(numpy.float16),
                "conv_process": layer["conv_process"].astype(numpy.float16),
                "upsample_factor": layer["upsample_factor"]
            } for layer in self.decoder_layers],
            "output_conv": self.output_conv.astype(numpy.float16)
        }

        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

    def step(self, mel_spec: numpy.ndarray, real_audio: numpy.ndarray, gen_learning_rate=0.0002, disc_learning_rate=0.0002) -> typing.Tuple[float, float, numpy.ndarray]:
        fake_audio = self.generate(mel_spec)

        def clip_gradients(grad, max_norm=1.0):
            grad_norm = numpy.linalg.norm(grad)

            if grad_norm > max_norm:
                grad = grad * (max_norm / grad_norm)
            
            return grad

        def disc_loss_fn():
            real_pred = self.discriminate(real_audio)
            fake_pred = self.discriminate(fake_audio)
            return -numpy.mean(numpy.log(real_pred + 1e-8) + numpy.log(1 - fake_pred + 1e-8))

        for layer in self.disc_layers:
            grad = self._compute_numerical_gradient(layer['conv'], disc_loss_fn)
            layer['conv'], velocity = self._update_weights(layer['conv'], grad, disc_learning_rate)
            setattr(self, '_velocity', velocity)

        grad = self._compute_numerical_gradient(self.disc_output, disc_loss_fn)
        self.disc_output, velocity = self._update_weights(self.disc_output, grad, disc_learning_rate)

        def gen_loss_fn():
            fake_pred = self.discriminate(self.generate(mel_spec))
            return -numpy.mean(numpy.log(fake_pred + 1e-8))

        for layer in self.encoder_layers:
            for key in ['conv_down', 'conv_process']:
                grad = self._compute_numerical_gradient(layer[key], gen_loss_fn)
                layer[key], velocity = self._update_weights(layer[key], grad, gen_learning_rate)

        for key in ['conv1', 'conv2']:
            grad = self._compute_numerical_gradient(self.bottleneck[key], gen_loss_fn)
            self.bottleneck[key], velocity = self._update_weights(self.bottleneck[key], grad, gen_learning_rate)

        for layer in self.decoder_layers:
            for key in ['conv_up', 'conv_process']:
                grad = self._compute_numerical_gradient(layer[key], gen_loss_fn)
                layer[key], velocity = self._update_weights(layer[key], grad, gen_learning_rate)

        grad = self._compute_numerical_gradient(self.output_conv, gen_loss_fn)
        self.output_conv, velocity = self._update_weights(self.output_conv, grad, gen_learning_rate)

        gen_loss = gen_loss_fn()
        disc_loss = disc_loss_fn()

        self.history['gen_losses'].append(float(gen_loss))
        self.history['disc_losses'].append(float(disc_loss))

        return gen_loss, disc_loss, fake_audio

    def train(self, mel_specs: numpy.ndarray, real_audios: numpy.ndarray, epochs: int, batch_size: int = 32, checkpoint_dir: str = "./checkpoints", checkpoint_frequency: int = 10) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        n_batches = len(mel_specs) // batch_size

        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0

            indices = numpy.random.permutation(len(mel_specs))
            mel_specs_shuffled = mel_specs[indices]
            real_audios_shuffled = real_audios[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                batch_mel_specs = mel_specs_shuffled[start_idx:end_idx]
                batch_real_audios = real_audios_shuffled[start_idx:end_idx]

                gen_loss, disc_loss, _ = self.step(batch_mel_specs, batch_real_audios)
                epoch_gen_loss += gen_loss
                epoch_disc_loss += disc_loss

            epoch_gen_loss /= n_batches
            epoch_disc_loss /= n_batches

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Generator Loss: {epoch_gen_loss:.4f}")
            print(f"Discriminator Loss: {epoch_disc_loss:.4f}")

            self.history['epochs'] += 1

            if (epoch + 1) % checkpoint_frequency == 0:
                epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_dir, exist_ok=True)

                generator_checkpoint = {
                    "encoder_layers": self.encoder_layers,
                    "bottleneck": self.bottleneck,
                    "decoder_layers": self.decoder_layers,
                    "output_conv": self.output_conv
                }
                with gzip.open(os.path.join(epoch_dir, "generator.checkpoint"), "wb") as f:
                    pickle.dump(generator_checkpoint, f)

                discriminator_checkpoint = {
                    "disc_layers": self.disc_layers,
                    "disc_output": self.disc_output
                }
                with gzip.open(os.path.join(epoch_dir, "discriminator.checkpoint"), "wb") as f:
                    pickle.dump(discriminator_checkpoint, f)

                loss_checkpoint = {
                    "history": self.history
                }
                with gzip.open(os.path.join(epoch_dir, "loss.checkpoint"), "wb") as f:
                    pickle.dump(loss_checkpoint, f)

                config = {
                    "model_config": {
                        "mel_channels": self.mel_channels,
                        "hidden_channels": self.hidden_channels,
                        "target_sample_rate": self.target_sample_rate,
                        "upsample_scales": self.upsample_scales
                    },
                    "training_config": {
                        "epochs": epochs,
                        "current_epoch": epoch + 1,
                        "batch_size": batch_size,
                        "checkpoint_frequency": checkpoint_frequency
                    },
                    "checkpoint_paths": {
                        "generator": "generator.checkpoint",
                        "discriminator": "discriminator.checkpoint",
                        "loss": "loss.checkpoint"
                    }
                }
                
                with open(os.path.join(epoch_dir, "config.yaml"), "w") as f:
                    yaml.dump(config, f)

    def from_mel(self, mel: numpy.ndarray, output_path: str = "output.wav") -> numpy.ndarray:
        """
        Reconstructs audio waveform from mel-spectrogram using the trained vocoder and saves it as WAV.
        
        Args:
            mel: Input mel-spectrogram (shape: [mel_channels, time] or [batch, mel_channels, time])
            output_path: Path where to save the WAV file (default: "output.wav")
        
        Returns:
            Reconstructed audio waveform
        """
        if len(mel.shape) == 2:
            mel = mel[numpy.newaxis, ...]
            
        audio = self.generate(mel)
        
        audio_int16 = numpy.int16(audio * 32767)

        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(44100)  # Sample rate
            wav_file.writeframes(audio_int16.tobytes())
            
        return audio