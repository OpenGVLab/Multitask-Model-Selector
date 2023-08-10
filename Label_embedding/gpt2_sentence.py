from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('gpt2-large')
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
    # print(caption_tmp)
    if len(caption_tmp) != 0:
        inputs = tokenizer(caption_tmp, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states.mean(dim = 1).detach().cpu().numpy()
        caption_embeddings.append(embeddings)
    # print(image_embeddings.shape)

caption_embeddings_npy = np.vstack(caption_embeddings)
np.save('flickr8k_captions_nonorm_gpt2.npy',caption_embeddings_npy)

