import torchvision.transforms as transforms

def get_transforms(IMAGE_HEIGHT=218, IMAGE_WIDTH=178):
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])
    return train_transform, validation_transform