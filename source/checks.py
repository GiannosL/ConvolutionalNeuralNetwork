import os, yaml


class Input_Handling:
    def __init__(self, yaml_path:str) -> None:
        self.model = None
        self.yaml_file = self.read_yaml(path=yaml_path)
        self.train_flag = self.training_check()
        self.n_epochs = self.get_epochs()
        self.output_directory = self.get_output_path()

    def read_yaml(self, path:str) -> dict:
        """
        doc
        """
        with open(path, "r") as stream:
            try:
                local_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return local_dict

    def training_check(self) -> bool:
        """
        doc
        """
        # check train option in YAML file
        if not "train" in self.yaml_file.keys():
            raise Exception("[train] option should be included in YAML file [\"yes\"/\"no\"].")
        
        # check training flag
        if self.yaml_file["train"] == "yes":
            self.check_for_data()
            return True
        elif (not self.yaml_file["train"]) or (self.yaml_file["train"] == "no"):
            # check if model to load exists.
            self.check_for_model()
            return False
        else:
            raise Exception("[train] option should be yes/no.")
    
    def check_for_data(self) -> None:
        """
        doc
        """
        # check train option in YAML file
        if not "training_data" in self.yaml_file.keys():
            raise Exception("[training_data] option not included...")
        
        # when model should already exist
        path = self.yaml_file["training_data"]
        is_model = os.path.isdir(os.path.expanduser(path))
        # if model does not exist throw error
        if is_model:
            self.training_data = path
        else:
            raise Exception(f"Training file-structure specified through 'training_data'. Path: {path}")
    
    def check_for_model(self) -> None:
        """
        doc
        """
        # check model option in YAML file
        if not "save_model" in self.yaml_file.keys():
            raise Exception("[save_model] option not included...")
        
        # when model should already exist
        path = self.yaml_file["save_model"]
        is_model = os.path.exists(path)
        # if model does not exist throw error
        if is_model:
            self.model = path
        else:
            raise Exception(f"pytorch model specified through 'save_model'. Path: {path}")
    
    def get_output_path(self) -> str:
        """
        doc
        """
        # check model option in YAML file
        if not "output_directory" in self.yaml_file.keys():
            raise Exception("[output_directory] option not included...")
        
        # when model should already exist
        path = self.yaml_file["output_directory"]
        is_model = os.path.exists(path)

        # if model does not exist throw error
        if not is_model:
            raise Exception(f"Output directory 'output_directory' missing. Path: {path}")
        
        return path


    def get_epochs(self) -> None:
        """
        doc
        """
        # default value
        n_epochs = 10

        # check train option in YAML file
        if "epochs" in self.yaml_file.keys():
            n_epochs = int(self.yaml_file["epochs"])
        else:
            print("Option \"epochs\" is not set from YAML file. Default: 10")
            
        return n_epochs
        