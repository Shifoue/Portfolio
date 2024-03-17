import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json

import pandas as pd

seed = 42

def get_transforms(size=640):
    train_transforms = torch.nn.Sequential(
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    )

    test_transforms = torch.nn.Sequential(
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Grayscale(),
    )

def process_data(path_to_images, path_to_data_json):
    f_data = open(path_to_data_json)
    json_data = json.load(f_data)

    #Create DataFrame using json
    df_images = pd.DataFrame.from_dict(json_data["images"])
    df_annotations = pd.DataFrame.from_dict(json_data["annotations"])

    #Select sub DataFrame from images corresponding to annotated images
    new_df = df_images[df_images["id"].isin(df_annotations["image_id"])]
    new_df = new_df.rename(columns = {"id": "image_id"})

    #Merge dataset on common ID to get image name
    processed_df = pd.merge(new_df, df_annotations, on='image_id', how='outer')

    processed_data = list(processed_df[["file_name", "caption"]].itertuples(index=False, name=None))

    return [(os.path.join(path_to_images, x), y) for x,y in processed_data]