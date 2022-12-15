import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class Images:
    def __init__(self, directory_path:str, file_type:str="jpg") -> None:
        # path to image files
        self.file_paths = self.collect_files(directory_path, file_type)
        # image transformer
        self.transforms = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.other_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4)
        ])
        
        # images dimensions
        self.images = self.get_images()
    
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

        # look through the images in your collection
        for f in self.file_paths[:10]:
            with Image.open(f) as img:
                img = self.transforms(img)
                img = np.transpose(img.numpy(), (1,2,0))
                images.append(img)
                image_dimensions.append(img.size)
        
        # convert list to dataframe
        image_dim_df = pd.DataFrame(image_dimensions)
        image_dim_df["image"] = images
        image_dim_df.rename(columns={0: "X", 1: "Y"}, inplace=True)
        
        return image_dim_df
    
    def show(self, index:int) -> None:
        """
        show image object
        """
        # make sure index is within bounds
        assert(index <= self.images.shape[0])

        # open and show image
        plt.imshow(self.images["image"][index])
        plt.show()
