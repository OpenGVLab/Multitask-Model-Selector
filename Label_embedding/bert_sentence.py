import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np
# 指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
import transformers
from transformers import BertTokenizer, BertModel
import torch

# 加载tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-uncased")

# 加载模型
model = BertModel.from_pretrained('bert-large-uncased')
model.to(device)

train_ori = '/data/flickr8k/all.txt'
captions = []

with open(train_ori, "r") as file:
    for line in file:
        img, caption = line.strip().split(" ", 1)  # split each line into two values
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
    if len(caption_tmp) != 0:
        inputs = tokenizer(caption_tmp, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states.mean(dim = 1).detach().cpu().numpy()
        caption_embeddings.append(embeddings)
    # print(image_embeddings.shape)

caption_embeddings_npy = np.vstack(caption_embeddings)
np.save('flickr8k_captions_nonorm_bert.npy',caption_embeddings_npy)
