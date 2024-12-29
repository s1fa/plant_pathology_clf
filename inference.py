import torch
from torchvision import transforms
import gradio as gr
from model import PPClassifier, PPClassifierConfig
# from utils4training import get_dataset, get_transform

# CONSTANTS
labels2names = {
    0: 'complex',
    1: 'frog_eye_leaf_spot',
    2: 'healthy',
    3: 'powdery_mildew',
    4: 'rust',
    5: 'scab'
}
names2labels = {v: k for k, v in labels2names.items()}

# load model
weight_path = 'output/final_model_state_dict.pth'
config = PPClassifierConfig()
model = PPClassifier(config)
model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

# preprocess the image for model
def preprocess(img):
    trans_rule = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4870, 0.6265, 0.4082],
                             std=[0.1668, 0.1456, 0.1742])
    ])
    img = trans_rule(img)
    return img

# Build app
def inference(img, device='auto'):
    global model
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    model = model.to(device)
    model.eval()
    img = preprocess(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    confidences = {labels2names[i]: probs[i] for i in range(6)}
    return confidences

with gr.Blocks(theme=gr.themes.Soft(),
               title='Plant Pathology Classifier') as demo:
    gr.Markdown(
        '''
        # Plant Pathology Classifier
        '''
    )
    img = gr.Image(sources=['upload', 'clipboard'],
                         type='pil',
                         label='Input Image')
    label = gr.Label()
    device = gr.Radio(['auto', 'cpu', 'cuda', 'mps'],
                        label='Device')
    btn = gr.Button('Classify')
    btn.click(inference, [img, device], label)

demo.launch(server_name='0.0.0.0', server_port=7860)