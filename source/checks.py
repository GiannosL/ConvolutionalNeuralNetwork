import yaml


class Input_Handling:
    def __init__(self, yaml_path:str) -> None:
        self.yaml_file = self.read_yaml(path=yaml_path)
        self.train_flag = self.training_check()

    def read_yaml(self, path:str) -> dict:
        with open(path, "r") as stream:
            try:
                local_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return local_dict

    def training_check(self) -> bool:
        if self.yaml_file["train"] == "yes":
            return True
        elif (not self.yaml_file["train"]) or (self.yaml_file["train"] == "no"):
            return False
        else:
            raise("[train] option should be yes/no.")
