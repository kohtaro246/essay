import copy
import os
import joblib
import time

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

from common import set_seed
from dataset import ClassificationDataset
from model import ClassificationModel, MeanPooling

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

CONFIG = {
    "seed": 42,
    "epochs": 5,
    "model_name": "microsoft/deberta-v3-base",
    "n_accumulate": 2,
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "max_length": 512,
    "learning_rate": 3e-5,
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 0.01,
    "num_classes": 3,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
  }

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])

TRAIN_DIR = "/home/mil/k-tanaka/essay_feedback/essay_feedback/data/train"
TEST_DIR = "/home/mil/k-tanaka/essay_feedback/essay_feedback/data/test"

set_seed(CONFIG['seed'])

encoder = LabelEncoder()
train_df['discourse_effectiveness_label'] = encoder.fit_transform(train_df['discourse_effectiveness'])

with open("le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)

def train_one_epoch(model, optimizer, scheduler, dataloader, device):
    model.train()

    total = 0
    running_loss = 0.0
    correct = 0
    lr = []
    bar = tqdm(dataloader, total=len(dataloader))
    steps = len(dataloader)
    for step, data in enumerate(bar):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.long)
        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
        if (step + 1) % CONFIG['n_accumulate'] == 0 or step == steps:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        running_loss += loss.item() * batch_size
        total += batch_size
        _, predictions = outputs.max(1)
        correct += (predictions == targets).float().sum().item()
        epoch_loss = running_loss / total
        acc = correct / total
        bar.set_postfix(Loss=epoch_loss, Accuracy=acc*100, LR=optimizer.param_groups[0]['lr'])
        lr.append(optimizer.param_groups[0]['lr'])
    return epoch_loss, acc, lr

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    running_loss = 0.0
    correct = 0

    for data in dataloader:
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        total += batch_size

        _, predictions = outputs.max(1)
        correct += (predictions == targets).float().sum().item()

    epoch_loss = running_loss / total
    acc = correct / total

    print("Validation Loss: {:.4f} Accuracy: {:.2f}%".format(epoch_loss, acc * 100))
    return epoch_loss, acc

def start_training(model, optimizer, scheduler, device, num_epochs):
    start = time.time()
    best_epoch_loss = np.inf
    history = {"Train Loss": [], "Valid Loss": [], "Train Acc": [], "Valid Acc": [], "LR": []}

    for epoch in range(1, num_epochs + 1):
        print("Epoch: ", epoch)
        train_epoch_loss, train_epoch_acc, epoch_lr = train_one_epoch(
            model, optimizer, scheduler, dataloader=train_loader, device=CONFIG["device"]
        )

        val_epoch_loss, valid_epoch_acc = evaluate(
            model, valid_loader, device=CONFIG["device"]
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)
        history["Train Acc"].append(train_epoch_acc)
        history["Valid Acc"].append(valid_epoch_acc)
        history["LR"].extend(epoch_lr)

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(
                f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            best_epoch_loss = val_epoch_loss
            best_epoch_acc = valid_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_feedback.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )

    print(
        "Best Loss: {:.4f} Best Accuracy: {:.2f}".format(
            best_epoch_loss, best_epoch_acc * 100
        )
    )

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history

df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=42, stratify = train_df.discourse_effectiveness_label)
train_dataset = ClassificationDataset(df_train, tokenizer=CONFIG["tokenizer"], max_length=CONFIG["max_length"], data_path=TRAIN_DIR)
valid_dataset = ClassificationDataset(df_valid, tokenizer=CONFIG["tokenizer"], max_length=CONFIG["max_length"], data_path=TRAIN_DIR)

collate_fn = DataCollatorWithPadding(tokenizer=CONFIG['tokenizer'])
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["train_batch_size"],
    collate_fn=collate_fn,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG["valid_batch_size"],
    collate_fn=collate_fn,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)

model = ClassificationModel(CONFIG['model_name'],CONFIG['num_classes'])
model.to(CONFIG['device'])

optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

w_adequate = 1-len(train_df[train_df['discourse_effectiveness'] == 'adequate'])/len(train_df)
w_effective = 1-len(train_df[train_df['discourse_effectiveness'] == 'effective'])/len(train_df)
w_ineffective = 1-len(train_df[train_df['discourse_effectiveness'] == 'ineffective'])/len(train_df)
class_weights = torch.tensor(
    [w_adequate, w_effective, w_ineffective]
).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)

model, history = start_training(model, optimizer, scheduler, device=CONFIG['device'], num_epochs=CONFIG['epochs'])

