# from torchvision.models import AlexNet_Weights # FIXXX
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def expand_dataset(folder):
    " Returns original dataset with additional augmented versions of the set"

    # Original dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    dataset0 = ImageFolder(root=folder, transform=transform)

    # Augmented dataset 1
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.ColorJitter(brightness=0.5),
    #     transforms.RandomAffine(
    #         degrees=(10, 80), translate=(0.1, 0.3), scale=(0.75, 1)),
    #     transforms.Resize((256, 256)),
    # ])

    # dataset1 = ImageFolder(root=folder, transform=transform)

    # Augmented dataset 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(256),
    ])

    dataset2 = ImageFolder(root=folder, transform=transform)

    # Augmented dataset 3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomPerspective(),
        transforms.Resize((256, 256)),
    ])

    dataset3 = ImageFolder(root=folder, transform=transform)

    # Augmented dataset 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomPerspective(),
        transforms.RandomAffine(degrees=(0, 180)),
        transforms.Resize((256, 256)),
    ])

    dataset4 = ImageFolder(root=folder, transform=transform)

    return torch.utils.data.ConcatDataset(
        [dataset0, dataset2, dataset3, dataset4])


def visualize_activation_space(activations, layer, cav=None):
    " Visualize the activation space of the various data classes (using t-SNE), for a given layer. Option to include and visualize layer CAV"

    # Generate array of all activations
    all_activations = None
    labels = []
    for class_name in activations:
        if all_activations is None:
            all_activations = activations[class_name][str(layer)]
        else:
            all_activations = np.concatenate(
                (all_activations, activations[class_name][str(layer)]), axis=0)

        # Track activations classes, for coloring in visualization
        labels += [int(class_name)] * len(activations[class_name][str(layer)])

    # Flatten activations
    flat = all_activations.reshape(all_activations.shape[0], -1)

    # Include CAV vector if present
    if not (cav is None):
        flat = np.concatenate((flat, np.array([cav[str(layer)]])), axis=0)
        labels.append(int(class_name) + 1)

    # Train t-SNE
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(flat)

    # Create dataframe for reduced points
    result = pd.DataFrame(tsne_results, columns=['x', 'y'])
    result['labels'] = labels

    # Plot reduced activations, with classes defining colors
    ax = plt.axes()
    ax.scatter(result.x, result.y, c=result.labels)


def get_train_loaders(train_dataset, batch_size=16):
    " Generate training and validation loaders from a given training set"

    # Split dataset
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_dataset.targets)),
        test_size=0.2,
        shuffle=True,
        stratify=train_dataset.targets)

    # Generate split indexes
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # Generate dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def change_sensitive_labels(train_dataset, sensitive_indexes):
    " Change labels in training set for data which are sensitive to a specific concept"
    train_dataset.samples = [(d, 1) if i in sensitive_indexes else (
        d, s) for i, (d, s) in enumerate(train_dataset.samples)]
    return train_dataset
