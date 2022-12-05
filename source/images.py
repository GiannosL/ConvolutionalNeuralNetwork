from glob import glob


class Images:
    def __init__(self, directory_path:str, file_type:str="jpg") -> None:
        #
        self.file_paths = self.collect_files(directory_path, file_type)
    
    def collect_files(self, directory_path:str, file_type:str):
        """
        collect file paths
        """
        # check directory path format
        if directory_path.endswith("/"):
            directory_path = directory_path[:-1]
        
        files = glob(f"{directory_path}/*.{file_type}")
        return files
