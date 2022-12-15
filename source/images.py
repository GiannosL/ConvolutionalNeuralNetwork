import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as torch_transforms


class Images:
    def __init__(self, directory_path:str, file_type:str="jpg") -> None:
        # path to image files
        self.file_paths = self.collect_files(directory_path, file_type)
        
        # images dimensions
        self.metadata, self.images = self.get_images()
    
    def collect_files(self, directory_path:str, file_type:str) -> list:
        """
        collect file paths
        """
        # check directory path format
        if directory_path.endswith("/"):
            directory_path = directory_path[:-1]
        # collect file paths
        files = glob(f"{directory_path}/*.{file_type}")

        return files
    
    def get_images(self) -> pd.DataFrame:
        """
        collect images from their file paths
        """
        image_dimensions = []
        images = []

        # parse through the images to get dimensions
        for f in self.file_paths[:10]:
            with Image.open(f) as img:
                image_dimensions.append(img.size)
        
        image_metadata = pd.DataFrame(image_dimensions)
        image_metadata.rename(columns={0: "X", 1: "Y"}, inplace=True)
        max_dims = image_metadata.X.max() if image_metadata.X.max() >= image_metadata.Y.max() else image_metadata.Y.max()

        # create image transformer
        transforms = torch_transforms.Compose([
            torch_transforms.Resize((max_dims, max_dims)),
            torch_transforms.CenterCrop(max_dims),
            torch_transforms.RandomHorizontalFlip(p=0.4),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # save images as tensors, give them maximum dimensions
        for f in self.file_paths[:10]:
            with Image.open(f) as img:
                img = transforms(img)
                img = np.transpose(img.numpy(), (1,2,0))
                images.append(img)
        
        # convert list to dataframe
        images = torch.Tensor(images)
        
        return image_metadata, images
    
    def show(self, index:int) -> None:
        """
        show image object
        """
        # make sure index is within bounds
        assert(index <= self.images.shape[0])

        # open and show image
        plt.imshow(self.images[index])
        plt.show()
