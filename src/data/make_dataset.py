import pandas as pd
import numpy as np
import gc
import feather

from sklearn.preprocessing import LabelEncoder


class Dataset:

    def __init__(self, path):
        self.path = path

    def load_files(self):
        self.train = pd.read_csv('%s/train.csv'%(self.path))
        self.test  = pd.read_csv('%s/test.csv'%(self.path))
        self.sub   = pd.read_csv('%s/sample_submission.csv'%(self.path))

        return self

    def encode_target(self):
        target = self.train['SignFacing (Target)']

        lbl = LabelEncoder()
        lbl.fit(target)

        self.train['SignFacing (Target)'] = lbl.transform(target)

        return self

    def rename_target(self):
        self.train = self.train.rename(columns={
                'SignFacing (Target)': 'Target'
            })

        return self

    def get_train_mask(self):
        return self.data.Target.notnull()

    def concat_data(self):
        self.data = pd.concat((self.train, self.test))
        self.train_mask = self.get_train_mask()

        del self.train, self.test
        gc.collect()

        return self


    def save_data(self, path):
        feather.write_dataframe(self.data, path)
        return self

    def load_data(self, path):
        feather.read_dataframe(path)
        return self