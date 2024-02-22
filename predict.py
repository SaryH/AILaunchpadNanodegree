import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=False)  # Initialize the model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)  # Reconstruct the final layer

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        
    # Use softmax to convert logits to probabilities
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_classes = probabilities.topk(topk, dim=1)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each.item()] for each in top_classes[0]]
    top_probs = top_probs[0].cpu().numpy()

    return top_probs, top_classes

# Example function for making predictions
def prediction(image_path, checkpoint, top_k, category_names, gpu):
    model, optimizer, epochs = load_checkpoint(checkpoint)
    
    image = process_image(image_path)

    device = None
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA...")
    else:
        device = 'cpu'
        print("Using CPU...")
    # Make the prediction
    probs, classes = predict(image_path, model, topk=top_k device=device)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Converting the classes to names
    names = [cat_to_name[str(cls)] for cls in classes]

    # Display the image
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=1, nrows=2)
    ax1.axis('off')
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title(names[0])

    # Display the topk classes
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name')
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top KK most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    # Call the predict function with the parsed arguments
    prediction(args.input, args.checkpoint, args.top_k, args.category_names, args.gpu)