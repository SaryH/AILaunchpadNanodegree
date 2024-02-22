import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu, batch_size):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = None
    optimizer = None
    
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)

        # Freezing all model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Replacing the last fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)  # 102 classes for the flower dataset
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif arch == 'vgg16':
        # Load a pre-trained VGG-16 network
        model = models.vgg16(pretrained=True)

        # Freeze the parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 1024),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(1024, 102),
                           nn.LogSoftmax(dim=1))

        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        
    device = None
    # Move model to the GPU if available
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA...")
    else:
        device = 'cpu'
        print("Using CPU...")
    
    device = torch.device(device)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()    
    
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {running_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}, "
                      f"Valid accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
                
                model.eval()

                accuracy = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)

                        # Get the class probabilities
                        ps = torch.exp(outputs)

                        # Get the top class of the output
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)

                        # Calculate accuracy by comparing to true labels
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        total += 1

    print(f"Test accuracy: {accuracy/total*100}%")

    model.class_to_idx = train_dataset.class_to_idx

    # Creating checkpoint dictionary
    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx,  # class to index map
        'state_dict': model.state_dict(),  # model state
        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
        'epochs': epochs,  # number of epochs
    }

    # saving the checkpoint
    torch.save(checkpoint, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['vgg16', 'resnet50'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training (default: 128)')

    args = parser.parse_args()
    
    # Call the training function with the parsed arguments
    train_model(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu, args.batch_size)
