import torch
from torch.utils.data import DataLoader
from transformers import pipeline
from utils import *



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


predict(dataloader,audio_classifier,templ,index)























