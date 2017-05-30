from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from tqdm import tqdm_notebook

def get_train_test_split(X, y, **params):
	X_train, X_test, y_train, y_test = train_test_split(X, y, **params)

	return X_train, X_test, y_train, y_test

def cross_validation(X, y, model, seed):
	skf = StratifiedKFold(n_splits=3, random_state=seed)

	scores = []

	for (itr, ite) in tqdm_notebook(skf.split(X, y)):
		
		Xtr = X.iloc[itr]
		ytr = y.iloc[itr]

		Xte = X.iloc[ite]
		yte = y.iloc[ite]

		model.fit(Xtr, ytr)

		fold_preds = model.predict_proba(Xte)
		fold_ll = log_loss(yte, fold_preds)

		scores.append(fold_ll)

	return scores
