import math
from pathlib import Path
import torch
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss
import datasets

# ----------------- Constants ----------------------
labels2names = {
    0: 'complex',
    1: 'frog_eye_leaf_spot',
    2: 'healthy',
    3: 'powdery_mildew',
    4: 'rust',
    5: 'scab'
}
names2labels = {v: k for k, v in labels2names.items()}
# ---------------------------------------------------
# data
def get_dataset(data_path: str):
    train_files = list(Path(data_path).glob('train*'))
    train_files = sorted([train_file.name for train_file in train_files])
    validation_files = list(Path(data_path).glob('validation*'))
    validation_files = sorted([validation_file.name for validation_file in validation_files])
    ds = datasets.load_dataset(
        data_path,
        data_files={
            'train': train_files,
            'validation': validation_files
        }
    )
    train_ds = ds['train']
    validation_ds = ds['validation']
    return train_ds, validation_ds

trans_rule = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4870, 0.6265, 0.4082],
                         std=[0.1668, 0.1456, 0.1742])
])
def get_transform(trans_rule=trans_rule):
    def trans(batch):
        return {
            'pixel_values': [trans_rule(img) for img in batch['image']],
            'encoded_labels': torch.tensor([[1 if i in labels else 0
                                             for i in range(6)]
                                             for labels in batch['labels']],
                                             dtype=torch.float32)
        }
    return trans

# model
def get_lr(step, warmup_steps, lr_decay_steps, max_lr, min_lr):
    # 1) linear warmup for warmup steps
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    # 2) constant learning rate for lr_decay_steps
    if step > lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# eval
def estimate_loss(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    losses = torch.zeros(len(dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            encoded_labels = batch['encoded_labels'].to(device)
            logits = model(pixel_values)
            cross_entropy = BCEWithLogitsLoss()
            loss = cross_entropy(logits, encoded_labels)
            losses[i] = loss.item()
    model.train()
    return losses.mean()