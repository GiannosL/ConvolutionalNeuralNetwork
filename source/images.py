from glob import glob


class Images:
    def __init__(self, directory_path:str, file_type:str="jpg") -> None:
        # path to image files
        self.file_paths = self.collect_files(directory_path, file_type)
        # images
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
    
    def get_images(self):
        pass
