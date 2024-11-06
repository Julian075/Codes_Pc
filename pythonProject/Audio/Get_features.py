from utils import *
from transformers import ClapModel, ClapProcessor
import os

#CLAP model
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
max_length=3
data_path='/home/julian/PycharmProjects/pythonProject/datos/Kale/data/'

audio_embeddings={'train':{},'val':{},'test':{}}
text_embeddings={}
for folder_type in os.listdir(data_path):
    for category in os.listdir(os.path.join(data_path,folder_type)):
        if not (category in text_embeddings.keys()):
            text_embeddings[category]=biolingual_features_text(f'A sound of {category}',model,processor,device)
        for name_audio in os.listdir(os.path.join(data_path,folder_type,category)):
            audio_path=os.path.join(data_path,folder_type,category,name_audio)
            audio_embeddings[folder_type][name_audio]=(category,biolingual_features_audio(audio_path,model,processor,max_length,device))




# Save dictionaries to a .pt file
save_path = 'audio_text_embeddings.pt'
torch.save({'audio_embeddings': audio_embeddings, 'text_embeddings': text_embeddings}, save_path)
print(f"Embeddings saved to {save_path}")


