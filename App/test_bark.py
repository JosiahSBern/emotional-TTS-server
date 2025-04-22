from transformers import AutoProcessor, BarkModel
import torch
import numpy as np
import soundfile as sf
import os

class TTS:



# Load Bark
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Set voice preset
voice_preset = "v2/en_speaker_6"  # Feel free to try others

# Prepare input
inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

# Generate audio (may be slow on CPU)
with torch.no_grad():
    audio_array = model.generate(**inputs)

# Convert to NumPy
audio_array = audio_array.cpu().numpy().squeeze()

# Save as .wav
sf.write("bark_output.wav", audio_array, samplerate=22050)

# Play (Windows only)
os.system("start bark_output.wav")
