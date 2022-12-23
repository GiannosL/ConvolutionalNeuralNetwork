import os, yaml
from source.termcolors import Terminal_Colors as tm


class Input_Handling:
    def __init__(self, yaml_path:str) -> None:
        self.yaml_file = self.read_yaml(path=yaml_path)
        self.train_flag = self.training_check()
        self.n_epochs = self.get_epochs()
        self.output_directory = self.get_output_path()

        # generate output directory
        self.generate_output()

    def read_yaml(self, path:str) -> dict:
        """
        read yaml configuration file
        """
        with open(path, "r") as stream:
            try:
                local_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"{tm.fail}{exc}{tm.endc}")
        return local_dict

    def training_check(self) -> bool:
        """
        check that variables necessary for training
        exist in the yaml file
        """
        # check train option in YAML file
        if not "train" in self.yaml_file.keys():
            raise Exception("[train] option should be included in YAML file [\"yes\"/\"no\"].")
        
        # check training flag
        if self.yaml_file["train"] == "yes":
            self.check_for_data()
            self.set_model_name()
            return True
        elif (not self.yaml_file["train"]) or (self.yaml_file["train"] == "no"):
            # check if model to load exists.
            self.check_for_model()
            return False
        else:
            raise Exception("[train] option should be yes/no.")
    
    def check_for_data(self) -> None:
        """
        make sure the file structure for the 
        input data is set
        """
        # check train option in YAML file
        if not "training_data" in self.yaml_file.keys():
            raise Exception(f"{tm.fail}[training_data] option not included...{tm.endc}")
        
        # when model should already exist
        path = self.yaml_file["training_data"]
        is_model = os.path.isdir(os.path.expanduser(path))
        # if model does not exist throw error
        if is_model:
            self.training_data = path
        else:
            raise Exception(f"{tm.fail}Training file-structure specified through 'training_data'. Path: {path}{tm.endc}")
    
    def set_model_name(self) -> None:
        """
        set model name when training
        """
        # check model option in YAML file
        if "model_name" in self.yaml_file.keys():
            self.model_name = self.yaml_file["model_name"]
        else:
            self.model_name = "Pythagoras"
            
    def check_for_model(self) -> None:
        """
        look for the model in the specified path
        """
        # check model option in YAML file
        if not "model_name" in self.yaml_file.keys():
            raise Exception(f"{tm.fail}[model_name] option not included...{tm.endc}")
        
        # when model should already exist
        path = self.yaml_file["model_name"]
        is_model = os.path.exists(f"{path}.pt")
        # if model does not exist throw error
        if is_model:
            self.model_name = path
        else:
            raise Exception(f"{tm.fail}PyTorch model specified through 'model_name'. Path: {path}{tm.endc}")
    
    def get_output_path(self) -> str:
        """
        validate the existance of the output path
        """
        # check model option in YAML file
        if not "output_directory" in self.yaml_file.keys():
            raise Exception(f"{tm.fail}[output_directory] option not included...{tm.endc}")
        
        # when model should already exist
        path = os.path.expanduser(self.yaml_file["output_directory"]) 
        is_model = os.path.isdir(path)
        # if model does not exist throw error
        if not is_model:
            raise Exception(f"{tm.fail}Output directory 'output_directory' missing. Path: {path}{tm.endc}")
        
        return path[:-1]

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
            print(f"{tm.warning}Option \"epochs\" is not set from YAML file. Default: 10{tm.endc}")
            
        return n_epochs
    
    def generate_output(self) -> None:
        """
        generates directory structure 
        for output files
        """
        # directories needed
        self.plot_dir = f"{self.output_directory}/plots/"
        self.result_dir = f"{self.output_directory}/results/"
        self.model_dir = f"{self.output_directory}/models/"
        self.report_dir = f"{self.output_directory}/reports/"

        directory_list = [self.plot_dir, self.result_dir, self.model_dir, self.report_dir]
        
        for direc in directory_list:
            try:
                os.mkdir(direc)
            except:
                print(f"{tm.warning}Could not generate directory: {direc} !{tm.endc}")
        