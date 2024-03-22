import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json
import random

import pandas as pd

from COCO_dataset import COCODataset

random.seed(42)

def get_transforms(size=640):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    ])

    return train_transforms, test_transforms


def process_data(path_to_images, path_to_json_data):
    f_data = open(path_to_json_data)
    json_data = json.load(f_data)

    #Create DataFrame using json
    df_images = pd.DataFrame.from_dict(json_data["images"])
    df_annotations = pd.DataFrame.from_dict(json_data["annotations"])

    #Select sub DataFrame from images corresponding to annotated images
    new_df = df_images[df_images["id"].isin(df_annotations["image_id"])]
    new_df = new_df.rename(columns = {"id": "image_id"})

    #Merge dataset on common ID to get image name (JOIN like operation)
    processed_df = pd.merge(new_df, df_annotations, on='image_id', how='outer')

    processed_data = list(processed_df[["file_name", "caption"]].itertuples(index=False, name=None))

    return [(os.path.join(path_to_images, x), y) for x,y in processed_data]


def create_dataloader(train_informations, test_informations):
    """
    Input : 
        train_informations : tuple(path_to_train_images_folder, path_to_train_json_data)
        test_informations : tuple(path_to_test_images_folder, path_to_test_json_data)

    Output :
        train_dataloader : Loader for training images and annotations
        validation_dataloader : Loader for validation images and annotations
        test_dataloader : Loader for test images and annotations
    """

    train_set = process_data(train_informations[0], train_informations[1]) # tuple(path_to_image, annotation)
    X_train, X_validation, y_train, y_valiation = train_test_split([x[0] for x in train_set], [x[1] for x in train_set], test_size=0.33, random_state=42)


    train_set = zip(X_train, y_train)
    validation_set = zip(X_validation, y_valiation)
    test_set = process_data(test_informations[0], test_informations[1]) # tuple(path_to_image, annotation)

    train_transforms, test_transforms = get_transforms()

    train_dataset = COCODataset(train_set, transforms=train_transforms)
    validation_dataset = COCODataset(validation_set, transforms=test_transforms)
    test_dataset = COCODataset(test_set, transforms=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader