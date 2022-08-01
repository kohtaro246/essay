import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AdamW
from tqdm import tqdm
import os
import re

class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, data_path):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.discourse_text = df['discourse_text'].values
        self.essay_id = df['essay_id'].values
        self.discourse_type = df['discourse_type'].values
        self.targets = df['discourse_effectiveness_label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse_text = self.discourse_text[index]
        discourse_type = self.discourse_type[index]
        essay_path = os.path.join(
            self.data_path, f"{self.essay_id[index]}.txt")
        essay = open(essay_path, 'r').read()

        text = discourse_type + " " + self.tokenizer.sep_token + \
            discourse_text + " " + self.tokenizer.sep_token + " " + essay
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }

class MLMDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        tokenized_data = self.tokenizer.encode_plus(
                            text,
                            max_length = 512,
                            truncation = True,
                            padding = 'max_length',
                            add_special_tokens = True,
                            return_tensors = 'pt'
                        )
        input_ids = torch.flatten(tokenized_data.input_ids)
        attention_mask = torch.flatten(tokenized_data.attention_mask)
        labels = getMaskedLabels(input_ids, self.tokenizer)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def getMaskedLabels(input_ids, tokenizer):
    special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]',
                                        add_special_tokens = False,
                                        return_tensors='pt')
    special_tokens = torch.flatten(special_tokens["input_ids"])
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < 0.15)
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in special_tokens:
        token = special_token.item()
        mask_arr *= (input_ids != token)
    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    input_ids[selection] = 128000
    
    return input_ids

def normalise_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub("\n", " ", text)
    return text

def get_essay(essay_id):
    essay_path = os.path.join("/home/mil/k-tanaka/essay_feedback/essay_feedback/data/train/", f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

def create_mlm_dataset(using="all"):
    if using == "effectiveness":
        df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/data/train.csv", usecols = ['essay_id', 'essay_text'])
        df['essay_text'] = df['essay_id'].apply(get_essay)
        df['essay_text'] = df['essay_text'].apply(normalise_text)
        essay_data = df["essay_text"].to_numpy()
    elif using == "asap":
        train_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        valid_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/valid_set.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        test_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/test_set.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        df = pd.concat([train_df, valid_df, test_df], axis=0)
        df['essay_text'] = df['essay'].apply(normalise_text)
        essay_data = df["essay_text"].to_numpy()
    else:
        df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/data/train.csv")
        df['essay_text'] = df['essay_id'].apply(get_essay)
        df['essay_text'] = df['essay_text'].apply(normalise_text)
        effectiveness_essay_data = df["essay_text"].to_numpy()

        train_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        valid_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/valid_set.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        test_df = pd.read_csv("/home/mil/k-tanaka/essay_feedback/essay_feedback/asap_data/test_set.tsv", sep='\t', encoding='ISO-8859-1', usecols = ['essay']).dropna(axis=1)
        df = pd.concat([train_df, valid_df, test_df], axis=0)
        df['essay_text'] = df['essay'].apply(normalise_text)
        asap_essay_data = df["essay_text"].to_numpy()
        essay_data = np.concatenate([effectiveness_essay_data, asap_essay_data])

    return essay_data


if __name__ == '__main__':
    essay_data = create_mlm_dataset("all")
