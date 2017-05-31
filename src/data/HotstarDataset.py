import pandas as pd
import numpy as np

import json
import gc
import feather

from sklearn.preprocessing import LabelEncoder

class Hotstar:

	def __init__(self, path):
		self.path = path

	def load_json(self):

		with open(self.path + 'train_data.json', 'r') as infile:
			train_json = json.load(infile)
			self.train      = pd.DataFrame.from_dict(train_json, orient='index')
			
			self.train.reset_index(level=0, inplace=True)
			self.train.rename(columns = {'index':'ID'},inplace=True)
			
			del train_json
			gc.collect()

			infile.close()
			
		with open(self.path + 'test_data.json') as infile:
			test_json = json.load(infile)
			
			self.test = pd.DataFrame.from_dict(test_json, orient='index')
			self.test.reset_index(level=0, inplace=True)
			self.test.rename(columns = {'index':'ID'},inplace=True)
			
			del test_json
			gc.collect()

			infile.close()

		return self

	def encode_target(self):
		lbl = LabelEncoder()
		lbl.fit(self.train['segment'])

		self.train['segment'] = lbl.transform(self.train['segment'])

		return self

	def concat_data(self):
		self.data = pd.concat((self.train, self.test))

		del self.train, self.test
		gc.collect()

		return self

	def get_train_mask(self):
		return self.data.segment.notnull()

	def save_data(self, path):
		feather.write_dataframe(self.data, path)

		return self

	def load_data(self, path):
		self.data = feather.read_dataframe(path)

		return self
