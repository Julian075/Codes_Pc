import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import numpy as np



## Create a mapping for captions based on the folder (species) name
def create_caption_from_species(species_name, mean_f_peak, mean_f_min, mean_f_max):
    return f"Species: {species_name}, Peak Frequency: {mean_f_peak}, Min Frequency: {mean_f_min}, Max Frequency: {mean_f_max}"

def templates(df,list_sp):
    templ = []
    target_index={}
    cont=0
    for index, row in df.iterrows():
        if row['species'] in list_sp:
            templ.append( create_caption_from_species(row['species'], row['mean_f_peak'], row['mean_f_min'], row['mean_f_max']))
            target_index[row['species']]=cont
            cont=cont+1
    return templ,target_index


def templates2(class_names):
    templ = []
    target_index={}
    cont=0
    for class_name in class_names:
        templ.append( f"A sound of {class_name}")
        target_index[class_name]=cont
        cont=cont+1
    return templ,target_index

def templates3(class_names):
    templ = []
    target_index={}
    cont=0
    for class_name in class_names:
        templ.append( f"{class_name}")
        target_index[class_name]=cont
        cont=cont+1
    return templ,target_index

def templates4(df,list_sp):
    templ = []
    target_index={}
    taxonomy={}
    cont=0
    for index, row in df.iterrows():
        if row['species'] in list_sp:
            if not( row['taxonomy'] in target_index):
                templ.append( f"A sound of {row['taxonomy']}")
                target_index[row['taxonomy']]=cont
                cont=cont+1
            if  not( row['species'] in taxonomy):
                taxonomy[row['species']]=row['taxonomy']

    return templ,target_index,taxonomy

class AudioCaptionDatasetFromFolder(Dataset):
    def __init__(self, root_dir, target_index=None, max_length=None, mode=None,taxonomy=None):
        """
        Args:
            root_dir (string): Directory with subfolders for each species, containing audio files.
            processor (ClapProcessor): The CLAP processor for audio.
            feature_info (dict): Optional dict to provide mean_f_peak, mean_f_min, mean_f_max for each species.
        """
        self.root_dir = root_dir
        self.audio_files = []
        self.targets = []
        self.index = target_index if target_index else {}
        self.max_length=max_length
        self.taxonomy=taxonomy

        # Traverse the folder structure and collect file paths and species names
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path): #and species_folder in species:
                for file_name in os.listdir(species_path):
                    if file_name.endswith(".wav"):  # Assuming audio files are in .wav format
                        self.audio_files.append(os.path.join(species_path, file_name))
                        if mode == 0:
                            self.targets.append(self.index[species_folder])
                        elif mode == 1:
                            self.targets.append(self.index[self.taxonomy[species_folder]])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        target = self.targets[idx]

        # Load the audio file
        audio, sample_rate = sf.read(audio_path)
        # Convert max_length from seconds to samples
        max_length_in_samples = int(self.max_length * sample_rate)

        # Pad or truncate the audio to the desired max_length in samples
        if len(audio) > max_length_in_samples:
            audio = audio[:max_length_in_samples]  # Truncate if too long
        elif len(audio) < max_length_in_samples:
            # Pad if too short
            padding = np.zeros(max_length_in_samples - len(audio))
            audio = np.concatenate((audio, padding))

        return audio, target

root_dir = "/home/julian/PycharmProjects/pythonProject/datos/Kale/caso3/test"

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_classifier = pipeline(task="zero-shot-audio-classification", model="davidrrobinson/BioLingual",device=device)
#audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general",device=device)
# List of species (classnames)
species = ["Leptodactylus_fragilis", "Alouatta_sp", "Dendropsophus_microcephalus", "Leptodactylus_fuscus","Patagioenas_cayennensis","Nyctidromus_albicollis","Crypturellus_soui"]
species2 = ["Leptodactylus_fragilis", "Alouatta_sp", "Dendropsophus_microcephalus", "Leptodactylus_fuscus"]

# Generate captions for each species
# templ,index = templates(pd.read_csv(csv_path),species)
templ, index = templates3(species2)
# templ,index,taxonomy = templates4(pd.read_csv(csv_path),species)

# Create the dataset
# dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index,max_length=10, mode=1,taxonomy=taxonomy)
dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index, max_length=10, mode=0)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

correct_predictions = 0
total_predictions = 0

for audio, target in dataloader:
    audio_array = []
    for i in range(len(audio)):
        audio_array.append(audio[i].numpy())  # Convert the audio sample to a NumPy array

    output = audio_classifier(audio_array, candidate_labels=templ)  # Zero-shot classification

    # Iterate over the batch
    for i in range(len(audio)):
        # Get the predicted label with the highest score
        predicted_label = max(output[i], key=lambda x: x['score'])['label']

        # predicted_label = output[i][0]['label']

        # Get the ground truth target label
        target_label = templ[target[i].item()]  # Convert the target index to the expected label

        # Compare the predicted label with the true target label
        if predicted_label == target_label:
            correct_predictions += 1
        total_predictions += 1
# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")























