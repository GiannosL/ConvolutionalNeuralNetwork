import os
from torchvision import datasets, transforms

class Image_Loader:
    def __init__(self, path:str) -> None:
        # tranformations
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_data = datasets.ImageFolder(os.path.join(path, "train"),
                                          transform=self.train_transforms)
        self.test_data = datasets.ImageFolder(os.path.join(path, "test"),
                                          transform=self.test_transforms)
