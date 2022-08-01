import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ClassificationModel, self).__init__()
        self.drop = nn.Dropout(p=0.05)

        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.mpool = MeanPooling()
        self.fc = nn.Sequential(nn.Linear(self.config.hidden_size, num_classes))
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,output_hidden_states=False)
        out = self.mpool(out.last_hidden_state, mask)
        out = self.drop(out)
        out = self.fc(out)
        return out