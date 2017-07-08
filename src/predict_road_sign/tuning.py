import pandas as pd 
import numpy as np
import gc

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import xgboost as xgb

SEED = 231313
np.random.seed(SEED)


target_map = {
	'Front': 0,
	'Rear' : 1,
	'Left' : 2,
	'Right': 3
}

def load_files():
	train = pd.read_csv('../../data/raw/4b699168-4-here_dataset/train.csv')
	test  = pd.read_csv('../../data/raw/4b699168-4-here_dataset/test.csv')
	sub   = pd.read_csv('../../data/raw/4b699168-4-here_dataset/sample_submission.csv')

	# rename target column

	train = train.rename(columns={
		'SignFacing (Target)': 'target'
	})

	data  = pd.concat((train, test))
	train_mask = data.target.notnull()

	del train, test
	gc.collect()

	data['DetectedCamera'] = data.DetectedCamera.map(target_map)

	# Features #
	# ======== #
	#
	# 1. Create count variable for sign aspect ratio.
	# 2. Create a boolean to check whether detected camera was facing either front or rear or not.

	data['detected_camera_flag']    = data.DetectedCamera.isin([0, 1]).astype(np.int)


	features = ['AngleOfSign', 'DetectedCamera', 'detected_camera_flag']

	X = data.loc[train_mask, features]
	y = data.loc[train_mask, 'target'].map(target_map)

	return X, y




def score(params):
	print('Training with params :')
	print(params)

	num_round           = int(params['n_estimators'])
	params['max_depth'] = int(params['max_depth'])

	del params['n_estimators']

	# 10-fold cross-validation scheme
	skf       = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
	cv_scores = []

	for index, (itr, ite) in enumerate(skf.split(X_train, y_train)):
		print('Fold: {}'.format(index))

		Xtr = X_train.iloc[itr]
		ytr = y_train.iloc[itr]

		Xte = X_train.iloc[ite]
		yte = y_train.iloc[ite]


		dtrain = xgb.DMatrix(Xtr, label=ytr)
		dvalid = xgb.DMatrix(Xte, label=yte)
		
		# watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
		model       = xgb.train(params, dtrain, num_round)
		predictions = model.predict(dvalid).reshape((Xte.shape[0], 4))
		score_       = log_loss(yte, predictions)
		print("\tScore {0}\n\n".format(score_))

		cv_scores.append(score_)

	print('Mean CV score: {}'.format(np.mean(cv_scores)))

	return {'loss': np.mean(cv_scores), 'status': STATUS_OK}

def optimize(trials):
	space = {
			 'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
			 'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
			 'max_depth' : hp.quniform('max_depth', 1, 13, 1),
			 'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
			 'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
			 'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
			 'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
			 'num_class' : 4,
			 'eval_metric': 'mlogloss',
			 'objective': 'multi:softprob',
			 'nthread' : 6,
			 'silent' : 1,
			 'seed': SEED
			 }

	best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

	print(best)


X, y = load_files()
print("Splitting data into train and valid ...\n\n")

X_train, X_test, y_train, y_test = train_test_split(
	X, y, stratify=y, test_size=0.2, random_state=SEED)

#Trials object where the history of search will be stored
trials = Trials()

optimize(trials)