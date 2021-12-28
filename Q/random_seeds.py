import random
import pandas as pd


class RandomSeeds:
    def __init__(self, seed=1):
        random.seed(seed)
        self.seed_range = (0, 1000000)
        self.seed_list = []

    def make_seeds(self, num=50):
        self.seed_list = [random.randint(self.seed_range[0], self.seed_range[1]) for i in range(num)]
        return self.seed_list

    def seeds_list_to_csv(self, Save_path, Name="seeds.csv"):
        data_list = pd.DataFrame({"seeds": self.seed_list})
        data_list.to_csv(Save_path + "/" + Name)

    def read_seeds_list(self, Save_path, Name="seeds.csv"):
        self.seed_list = pd.read_csv(Save_path + "/" + Name)['seeds'].values.tolist()
        return self.seed_list
