from utils import *
from transformers import ClapModel, ClapProcessor
prompt='A sound of a Leptodactylus fuscus'
audio_path='/home/julian/PycharmProjects/pythonProject/datos/Kale/data/train/Leptodactylus fuscus/21_SMA03126_20210611_200000.wav'

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
max_length=3

print(biolingual_features(prompt,audio_path,model,processor,max_length,device))