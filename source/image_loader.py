from torch.utils.data import DataLoader
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
        path = path if path.endswith("/") else f"{path}/"

        train_dt = datasets.ImageFolder(f"{path}train",
                                          transform=self.train_transforms)
        test_dt = datasets.ImageFolder(f"{path}test",
                                          transform=self.test_transforms)
                                          
        # actual dataset and labels
        self.train_data = DataLoader(train_dt, batch_size=100, shuffle=False)
        self.train_labels = train_dt.classes
        self.test_data = DataLoader(test_dt, batch_size=100, shuffle=False)
        self.test_labels = test_dt.classes
