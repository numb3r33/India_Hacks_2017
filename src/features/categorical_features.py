import pandas as pd
import numpy as np
import scipy as sp

def woe(train, test, y, features='all', information_value=False, inplace=False):
	if len(set(y)) != 2:
		raise ValueError('Transformation is only used for binary classification.')
	
	#write only categorical names to variable cat_columns
	cat_columns_train = sorted([i for i in train.columns if train[i].dtype == 'O'])
	cat_columns_test = sorted([i for i in test.columns if test[i].dtype == 'O'])
	
	if cat_columns_train != cat_columns_test:
		raise ValueError('Train and test sets must have the same columns')
	
	if features == 'all':
		features = cat_columns_train 
	else:
		if type(features) != list:
			raise ValueError(str(type(features)) + ' type passed in features. Only list are allowed.')
		if len(features) == 0:
			raise ValueError('You have chosen not enough features. The minimum number is one.')
			
		feat_not_in_data = [i for i in features if i not in cat_columns_train]
		
		if len(feat_not_in_data) > 0:
			raise ValueError(','.join(feat_not_in_data) + ' are not in data.')
	
	
	# calculate weights of evidance
	def cal_woe(x_train, x_test, name):
		good = np.sum(y == 0)
		bad = np.sum(y == 1)
		inf_v = 0
		
		for i in x_train.unique():
			count_bad = float(np.sum((y == 1) & (x_train == i))) / bad
			count_good = float(np.sum((y == 0) & (x_train == i))) / good
			x_train[x_train==i] = np.log(count_good / count_bad)
			x_test[x_test==i] = np.log(count_good / count_bad)
			inf_v += (count_good - count_bad) * np.log(count_good / count_bad)
		
		return x_train, x_test, inf_v
	
	
	inf_value = dict()
	if inplace:
		
		for i in features:
			train[i], test[i], inf_value[i] = cal_woe(train[i], test[i], i)
		
		if information_value:
			return pd.DataFrame(inf_value.items(), 
									  columns = ['Feature', 'Inf_value']).sort('Inf_value', ascending=False)
		
	else:
		
		new_train = train.copy()
		new_test = test.copy()
		for i in features:
			new_train[i], new_test[i], inf_value[i] = cal_woe(new_train[i], new_test[i], i)
		
		if information_value:
			return new_train, new_test, pd.DataFrame(inf_value.items(), 
									  columns = ['Feature', 'Inf_value']).sort('Inf_value', ascending=False)
		return new_train, new_test


def encode_ohe(feature):
	"""
	feature: Pandas Series
	"""

	return feature.str.replace('[:](\d+)', '').str.get_dummies(sep=',')

def prepare_ohe_variables(data, features):
	"""
	data: Pandas Dataframe
	features: List of features
	"""
	for f in features:
		data = pd.concat((data, encode_ohe(data[f])), axis='columns')

	return data


def count_feature(feature):
	return feature.str.replace('[:](\d+)', '').map(lambda x: len(x.split(',')))

def num_seconds_watched(feature):
	return feature.str.replace('.+[:]', '').map(lambda x: np.sum([int(z) for z in x.split(',')]))	