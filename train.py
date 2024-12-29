import os
from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import PPClassifier, PPClassifierConfig
from utils4training import get_dataset, get_transform, get_lr, estimate_loss

# ---------------- Hyperparameters --------------------
out_dir = 'output'
log_dir = 'log'

# for dataloader
batch_size = 64
data_path = './pp16_8k/data'
# for logging
log_interval = 20
eval_interval = 100
best_val_loss = 1e9
# for optimizer and scheduling 
max_lr = 0.01
min_lr = 0.001
max_steps = 1200 * 5
warmup_steps = 200

weight_decay = 0.1
betas = (0.9, 0.95)
grad_clip = 1.0

# DDP setting
backend = 'nccl'

# ----------------- Main ---------------------
init_process_group(backend=backend)
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)

master_process = ddp_rank == 0
seed_offset = ddp_rank

# only the master process do the logging
if master_process:
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)

torch.manual_seed(1213 + seed_offset)
torch.cuda.manual_seed(1213 + seed_offset)

# Load the dataset
train_ds, validation_ds = get_dataset(data_path)

train_ds.set_transform(get_transform())
validation_ds.set_transform(get_transform())

class PPDataset(Dataset):
    def __init__(self, train_ds):
        self.ds = train_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

sampler = DistributedSampler(PPDataset(train_ds))
train_dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, pin_memory=True)
validation_dataloader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

# model init
config = {
    'image_size': 224,
    'patch_size': 16,
    'num_classes': 6,
    'embed_dim': 768,
    'num_heads': 12,
    'num_layers': 6,
    'bias': False,
    'dropout': 0.0,
}
pp_config = PPClassifierConfig(**config)
model = PPClassifier(pp_config).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=betas, weight_decay=weight_decay)

# wrap the model with DDP
model = DDP(model, device_ids=[ddp_local_rank])

# calculate the epochs
samples_per_steps = batch_size * ddp_world_size
total_epochs = int(max_steps * samples_per_steps // len(train_ds) + 1)

# training loop
steps_run = 0
for epoch in range(total_epochs):
    sampler.set_epoch(epoch)
    model.train()

    for batch in train_dataloader:
        if steps_run > max_steps:
            break

        lr = get_lr(steps_run, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device) # 
        encoded_labels = batch['encoded_labels'].to(device) # 

        logits = model(pixel_values)
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(logits, encoded_labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        steps_run += 1
        if steps_run % log_interval == 0 and master_process:
            log_str = f'> step {steps_run}: batch_loss {loss.item():.4f} | norm {norm:.4f} | lr {lr:.6f}'
            print(log_str)
            with open(f'{log_dir}/log.txt', 'a') as f:
                f.write(f'{log_str}\n')
        if steps_run % eval_interval == 0 and master_process:
            val_loss = estimate_loss(model, validation_dataloader)
            log_str = f'>> step {steps_run}: val_loss {val_loss:.4f}'
            print(log_str)
            with open(f'{log_dir}/log.txt', 'a') as f:
                f.write(f'{log_str}\n')
            # save the best val_loss model
            if val_loss < best_val_loss and steps_run > warmup_steps:  # after warmup
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'step': steps_run,
                    'config': config
                }
                torch.save(checkpoint, f'{out_dir}/checkpoint.pth')

if master_process:
    torch.save(model.module.state_dict(), f'{out_dir}/final_model_state_dict.pth')
    print('Model saved.')
    print('Training finished.')

destroy_process_group()
