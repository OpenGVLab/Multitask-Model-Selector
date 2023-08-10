import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
# 指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

train_ori = '/data/flickr_8k/flickr_jsons/all.txt'
captions = []

with open(train_ori, "r") as file:
    for line in file:
        img, caption = line.strip().split(" ", 1)  # split each line into two values
        print(img)
        captions.append(caption.lower())

caption_embeddings = []
batch_size = 100
print(len(captions)//batch_size)
for i in range(len(captions)//batch_size + 1):
    if (i+1)*batch_size >= len(captions):
        end_index = len(captions)
    else:
        end_index = (i+1)*batch_size
    start_index = i * batch_size
    print((i+1)*batch_size)

    caption_tmp = captions[start_index : end_index]
    # print(caption_tmp)
    if len(caption_tmp) != 0:
        inputs = processor(caption_tmp, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs).detach().cpu().numpy()
            # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  
        caption_embeddings.append(embeddings)
    # print(image_embeddings.shape)

caption_embeddings_npy = np.vstack(caption_embeddings)
np.save('flickr8k_captions_nonorm_clip.npy',caption_embeddings_npy)
