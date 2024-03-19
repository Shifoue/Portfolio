import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from dataset import CatDogDataset

seed = 42

def get_transforms(size=640):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    ])

    return train_transforms, test_transforms

def data_from_path(dataset_path):
    images = []
    labels = []

    child_directories = next(os.walk(dataset_path))[1] #These will be the labels
    for label in child_directories:
        path = os.join(dataset_path, label)

        images.append(os.join(path, f[:-1]) for f in os.listdir(path) if os.isfile(os.join(path, f)))
        labels.append(label)

    return images, labels

def processing(dataset_path):
    training_path = os.join(dataset_path, "training_set")
    test_path = os.join(dataset_path, "test_set")

    training_images, training_labels = data_from_path(training_path)
    test_images, test_labels = data_from_path(test_path)

    X_train, X_validation, y_train, y_valiation = train_test_split(training_images, training_labels, test_size=0.33, random_state=42)

    training_set = zip(X_train, y_train)
    validation_set = zip(X_validation, y_valiation)
    test_set = zip(test_images, test_labels)

    train_transforms, test_transforms = get_transforms()

    training_data = CatDogDataset(training_set, transforms=train_transforms)
    validation_data = CatDogDataset(validation_set, transforms=train_transforms)
    test_data = CatDogDataset(test_set, transforms=test_transforms)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader