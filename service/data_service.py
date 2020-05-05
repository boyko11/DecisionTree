import numpy as np
import pandas as pd


class DataService:

    def __init__(self):
        np.set_printoptions(suppress=True)

    @staticmethod
    def get_data(file_name):

        data = pd.read_csv(file_name, header=None).to_numpy()
        return data[:, :-1], data[:, -1]

