import os

import torch
from torch.utils.data import DataLoader
from transformers import pipeline
import pandas as pd
from utils import *



root_dir = "/home/julian/PycharmProjects/pythonProject/datos/Kale/data/test"
path_info = "/home/julian/PycharmProjects/pythonProject/datos/Kale/species_frequency_summary.cvs"

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_classifier = pipeline(task="zero-shot-audio-classification", model="davidrrobinson/BioLingual",device=device)
#audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general",device=device)
# Generate captions for each species
# templ,index = templates(pd.read_csv(csv_path),species)
species=os.listdir(root_dir)
df_fre_info=pd.read_csv(path_info)
templ, index = templates(df_fre_info,species)
#templ, index = templates(species)
# templ,index,taxonomy = templates4(pd.read_csv(csv_path),species)

# Create the dataset
# dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index,max_length=10, mode=1,taxonomy=taxonomy)
dataset = AudioCaptionDatasetFromFolder(root_dir=root_dir, target_index=index, max_length=10, mode=0)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


predict(dataloader,audio_classifier,templ,index)























