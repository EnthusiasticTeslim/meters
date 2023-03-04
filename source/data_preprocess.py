import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path, sep="\t", header=None).dropna()

        print(data.head())

        self.data = data.to_numpy()

        return self.data

    def split_data(self, data):
        train_validation, test = train_test_split(data, test_size = 0.2, random_state=12)
        
        return train_validation, test
    
    def normalize_data(self, xtrain_data, xtest_data):

        scaler = StandardScaler()

        scaler.fit(xtrain_data)

        norm_xtrain_data = scaler.transform(xtrain_data)
        norm_xtest_data = scaler.transform(xtest_data)
        
        return norm_xtrain_data, norm_xtest_data
        