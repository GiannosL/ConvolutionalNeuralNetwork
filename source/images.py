import pandas as pd
from PIL import Image
from glob import glob


class Images:
    def __init__(self, directory_path:str, file_type:str="jpg") -> None:
        # path to image files
        self.file_paths = self.collect_files(directory_path, file_type)
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
        image_dimensions = []
        images = []

        # look through the images in your collection
        for f in self.file_paths:
            with Image.open(f) as img:
                images.append(img)
                image_dimensions.append(img.size)
        
        # convert list to dataframe
        image_dim_df = pd.DataFrame(image_dimensions)
        image_dim_df["IMG"] = images
        image_dim_df.rename(columns={0: "X", 1: "Y"}, inplace=True)
        
        return image_dim_df
