# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from cu_mask import process_binary_mask,tensor_to_image
from PIL import Image

from threading import Thread
from transformers import RobertaTokenizer,RobertaModel

def random_rot_flip(image, label,cu_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    cu_label = np.rot90(cu_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    cu_label = np.flip(cu_label, axis=axis).copy()
    return image, label,cu_label


def random_rotate(image, label,cu_label):
    angle = np.random.randint(-20, 20) 
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    cu_label = ndimage.rotate(cu_label, angle, order=0, reshape=False)
    return image, label,cu_label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,cu_label, text = sample['image'], sample['label'],sample['cu_label'], sample['text']
        image, label,cu_label = image.astype(np.uint8), label.astype(np.uint8),cu_label.astype(np.uint8)
        image, label, cu_label = F.to_pil_image(image), F.to_pil_image(label) ,F.to_pil_image(cu_label)   
        x, y = image.size
        if random.random() > 0.5:
            image, label, cu_label = random_rot_flip(image, label,cu_label)
        elif random.random() > 0.5:
            image, label,cu_label = random_rotate(image, label,cu_label)
        if x != 224 or y != 224:
            image = zoom(image, (224 / x, 224 / y), order=3)  # why not 3?
            label = zoom(label, (224 / x, 224 / y), order=0)
            cu_label = zoom(cu_label, (224 / x, 224 / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        cu_label = to_long_tensor(cu_label)
        text = torch.Tensor(text)

        sample = {'image': image, 'label': label,'cu_label': cu_label, 'text': text}
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images



class ImageToImage2D_Train(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.model_ready = False
        Thread(target=self._preload_model).start()
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol') 
        self.culabelcol_path = os.path.join(dataset_path,'cu-labelcol') 
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.cu_mask_list = os.listdir(self.culabelcol_path)   
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def _preload_model(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('/data/roberta-base/')
        self.roberta_model = RobertaModel.from_pretrained('/data/roberta-base/')
        self.roberta_model.eval()  
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        self.model_ready = True
    def __getitem__(self, idx):

        while not self.model_ready: 
            pass
        image_filename = self.images_list[idx] 
        mask_filename = image_filename[: -3] + "png"  

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)  
        mask = cv2.resize(mask, (self.image_size, self.image_size))   
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        cu_mask = cv2.imread(os.path.join(self.culabelcol_path, mask_filename), 0)  
        cu_mask = cv2.resize(cu_mask, (self.image_size, self.image_size))
        cu_mask[cu_mask <= 0] = 0
        cu_mask[cu_mask > 0] = 1
        #===========================

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        image, cu_mask = correct_dims(image, cu_mask)  

        text = self.rowtext[mask_filename]   
        with torch.no_grad():  
            inputs = self.tokenizer(
                text,
                padding='max_length',     
                truncation=True,        
                max_length=70,          
                return_tensors='pt'     
            )
            
            outputs = self.roberta_model(**inputs)  
            text_embedding = outputs.last_hidden_state.squeeze(0)  

        
        # print(text_embedding.shape)
        if text_embedding.shape[0] > 70:
            text_embedding = text_embedding[:70, :]
        elif text_embedding.shape[0] < 70:
            padding = torch.zeros((70 - text_embedding.shape[0], 768))
            text_embedding = torch.cat([text_embedding, padding], dim=0)
        
        sample = {'image': image, 'label': mask,'cu_label':cu_mask, 'text': text_embedding}  

        if self.joint_transform:
            sample = self.joint_transform(sample)
        return sample, image_filename
    
