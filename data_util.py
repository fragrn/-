import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_scheduler,BertModel,BertPreTrainedModel, BertLayer,BertConfig,ResNetModel,ResNetForImageClassification
from transformers.models.roberta.modeling_roberta import RobertaEncoder
import transformers
import torchvision.transforms as transforms


max_seq_length = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""负责标签与id的映射"""
label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}


"""Text-Only"""
class TextDataset(Dataset):
    def __init__(self, df, label_to_id, text_field="text", label_field="label"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.text_field = text_field
        self.label_field = label_field

    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        label = self.label_to_id[self.df.at[index, self.label_field]]

        return text, label

    def __len__(self):
        return self.df.shape[0]
    

"""Text-Image"""
class ResNetDataset(Dataset):
    def __init__(self, df, label_to_id, train=False, text_field="text", label_field="label", image_path_field="image_path"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field

        # ResNet-50 settings
        self.img_size = 224
        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


        self.train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
        )

        self.eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ]
        )


    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        label = self.label_to_id[self.df.at[index, self.label_field]]
        img_path = self.df.at[index, self.image_path_field]


        image = Image.open(img_path)
        if self.train:
          img = self.train_transform_func(image)
        else:
          img = self.eval_transform_func(image)

        return text, label, img

    def __len__(self):
        return self.df.shape[0]


"""Dataset for augment"""
class ResNetDataset_aug(Dataset):
    def __init__(self, df, label_to_id, train=False, text_field="text", label_field="label", image_path_field="image_path"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field

        # ResNet-50 settings
        self.img_size = 224
        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


        self.train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
        )

        self.eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ]
        )


    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        text_aug = str(self.df.at[index, 'text_aug'])
        label = self.label_to_id[self.df.at[index, self.label_field]]
        img_path = self.df.at[index, self.image_path_field]


        image = Image.open(img_path)
        if self.train:
          img = self.train_transform_func(image)
        else:
          img = self.eval_transform_func(image)

        return text, text_aug, label, img

    def __len__(self):
        return self.df.shape[0]




class ResNetDataset_image_only(Dataset):
    def __init__(self, df, label_to_id, train=False, text_field="text", label_field="label", image_path_field="image_path"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field

        # ResNet-50 settings
        self.img_size = 224
        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


        self.train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
        )

        self.eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ]
        )


    def __getitem__(self, index):
        # text = str(self.df.at[index, self.text_field])
        label = self.label_to_id[self.df.at[index, self.label_field]]
        img_path = self.df.at[index, self.image_path_field]


        image = Image.open(img_path)
        if self.train:
          img = self.train_transform_func(image)
        else:
          img = self.eval_transform_func(image)
        

        return label, img

    def __len__(self):
        return self.df.shape[0]


"""Dataset_for_prediction"""
class ResNetDataset_for_test(Dataset):
    def __init__(self, df, label_to_id, train=False, text_field="text", label_field="label", image_path_field="image_path"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field

        # ResNet-50 settings
        self.img_size = 224
        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


        self.train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
        )

        self.eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ]
        )


    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        # label = self.label_to_id[self.df.at[index, self.label_field]]
        img_path = self.df.at[index, self.image_path_field]


        image = Image.open(img_path)
        if self.train:
          img = self.train_transform_func(image)
        else:
          img = self.eval_transform_func(image)

        return text, img

    def __len__(self):
        return self.df.shape[0]





def dataloader_for_train(train, valid, batch_size, augment=False):
    if augment == False:    
        train_dataset = ResNetDataset(df=train, label_to_id=label_to_id, train=True, text_field='text', label_field='label', image_path_field='image_path')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        sampler=train_sampler)


        valid_dataset = ResNetDataset(df=valid, label_to_id=label_to_id, train=False, text_field='text', label_field='label', image_path_field='image_path')
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                sampler=valid_sampler)
    
        return train_dataloader, valid_dataloader
    
    else:
        train_dataset_aug = ResNetDataset_aug(df=train, label_to_id=label_to_id, train=True, text_field='text', label_field='label', image_path_field='image_path')
        train_sampler_aug = RandomSampler(train_dataset_aug)
        train_dataloader_aug = DataLoader(dataset=train_dataset_aug,
                    batch_size=batch_size,
                    sampler=train_sampler_aug)


        test_dataset_aug = ResNetDataset_aug(df=valid, label_to_id=label_to_id, train=False, text_field='text', label_field='label', image_path_field='image_path')
        test_sampler_aug = SequentialSampler(test_dataset_aug)
        test_dataloader_aug = DataLoader(dataset=test_dataset_aug,
                            batch_size=batch_size,
                            sampler=test_sampler_aug)
        
        return train_dataloader_aug, test_dataloader_aug

    

       
def dataloader_for_pre(df_for_test, batch_size):
    test_dataset = ResNetDataset_for_test(df=df_for_test, label_to_id=label_to_id, train=False, text_field='text', label_field='label', image_path_field='image_path')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset,
                    batch_size=batch_size,
                    sampler=test_sampler)
    
    return test_dataloader




   
   