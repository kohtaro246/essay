import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AdamW
from tqdm import tqdm
from dataset import MLMDataset, create_mlm_dataset
from common import set_seed
import os

import warnings
warnings.filterwarnings("ignore")

# config
seed = 42
model_name = 'microsoft/deberta-v3-base'
epochs = 3
batch_size = 8
lr = 1e-6
weight_decay = 1e-6
n_accumulate = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed
set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
#model = torch.nn.DataParallel(model)

essay_data = create_mlm_dataset()
dataset = MLMDataset(essay_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_loop(model, device):
    model.train()
    batch_losses = []
    loop = tqdm(dataloader, leave=True)
    for batch_num, batch in enumerate(loop):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        #print(torch.mean(loss))
        batch_loss = loss / n_accumulate
        batch_losses.append(batch_loss.item())
    
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=batch_loss.item())
        batch_loss.backward()
        if batch_num % n_accumulate == 0 or batch_num == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            model.zero_grad()

    return np.mean(batch_losses)

device = device
model.to(device)
history = []
best_loss = np.inf
prev_loss = np.inf
model.gradient_checkpointing_enable()
print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

for epoch in range(epochs):
    loss = train_loop(model, device)
    history.append(loss)
    print(f"Loss: {loss}")
    if loss < best_loss:
        print("New Best Loss {:.4f} -> {:.4f}, Saving Model".format(prev_loss, loss))
        torch.save(model.state_dict(), "./deberta_mlm.pt")
        best_loss = loss
    prev_loss = loss



