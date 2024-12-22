import torch
from model import PPClassifier, PPClassifierConfig
from utils4training import get_dataset, get_transform

def inference(model, img, device='auto'):
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    model = model.to(device)
    model.eval()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    return torch.sigmoid(logits).cpu()


weight_path = 'output/final_model_state_dict.pth'

config = PPClassifierConfig()
model = PPClassifier(config)
model.load_state_dict(torch.load(weight_path))

_, val_ds = get_dataset('pp16_8k/data')

idx = 10
print('True:', val_ds[idx]['labels'], val_ds[idx]['label_names'])

val_ds.set_transform(get_transform())
# inference
sample = val_ds[idx]
img = sample['pixel_values']
probs = inference(model, img)
# labels = sample['encoded_labels']

print('Predicted:', probs)