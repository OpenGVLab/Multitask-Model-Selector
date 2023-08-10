import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CaptionDataset(Dataset):
    def __init__(self, root_dir, annotations_file, feature_extractor, tokenizer,transform=None):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.imgs = []
        self.captions = []
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_sequence_length = 42
        self.do_padding = False if self.tokenizer.pad_token_id is None else True
        with open(self.annotations_file, 'r') as f:
            for line in f.readlines():
                img, caption = line.split(' ',1)
                
                caption = caption.strip()
                img_path = os.path.join(self.root_dir, img)
                self.imgs.append(img_path)
                self.captions.append(caption.lower())  #use uncased

        self.encoded = self.tokenizer(
                        self.captions,
                        padding="max_length" if self.do_padding else "do_not_pad",
                        max_length=self.max_sequence_length,
                        truncation=True,
                        return_tensors="np",
                        return_length=True
                        ).input_ids

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,index):
        img_path = self.imgs[index]
        caption = self.captions[index]
        pixel_values = self.feature_extractor(
            Image.open(img_path).convert("RGB"),
            return_tensors="np"
        ).pixel_values
        return pixel_values.squeeze(),self.encoded[index],list(self.encoded[index]).index(0)
        