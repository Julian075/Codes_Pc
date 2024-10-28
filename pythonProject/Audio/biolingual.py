import torch
import numpy as np
import librosa  # For loading audio files
from transformers import pipeline
import torchaudio
import os

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define audio file path (replace with yours)
audio_file = "/home/julian/PycharmProjects/pythonProject/datos/Kale/caso2/train/Alouatta_sp/2977_SMA03247_20210325_055000.wav"

# Check file existence (optional, for robustness)
if not os.path.exists(audio_file):
    print(f"Error: Audio file not found: {audio_file}")
    exit(1)  # Exit with an error code

# Load audio using librosa (consider error handling)
try:
    yr, sr = torchaudio.load(audio_file)
    # Verificar si el audio es estéreo
    if yr.shape[0] == 2:  # Si tiene 2 canales, es estéreo
        # Convertir a mono promediando los canales
        yr = torch.mean(yr, dim=0, keepdim=True)

    # Ensure audio is a NumPy array
    y = yr.numpy()
except Exception as e:
    print(f"Error loading audio: {e}")
    exit(1)  # Exit with an error code

# Ensure audio is a NumPy array
if not isinstance(y, np.ndarray):
    y = np.array(y)  # Convert if necessary

# Create audio classifier pipeline and move it to device
audio_classifier = pipeline(
    task="zero-shot-audio-classification", model="davidrrobinson/BioLingual",device=device
)

# Perform inference
print(camila.
with torch.no_grad():
    output = audio_classifier(y, candidate_labels=[
        "Sound of a sperm whale", "Sound of a sea lion"
    ])
    print(output)