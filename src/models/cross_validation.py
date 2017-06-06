from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

def get_train_test_split(X, y, **params):
	X_train, X_test, y_train, y_test = train_test_split(X, y, **params)

	return X_train, X_test, y_train, y_test

def cv_loop(X, y, model, metric_type, seed):
	skf = StratifiedKFold(n_splits=3, random_state=seed)

	scores = []

	for (itr, ite) in (skf.split(X, y)):
		
		Xtr = X.iloc[itr]
		ytr = y.iloc[itr]

		Xte = X.iloc[ite]
		yte = y.iloc[ite]

		model.fit(Xtr, ytr)

		fold_preds = model.predict_proba(Xte)

		if metric_type == 'log_loss':
			fold_score = log_loss(yte, fold_preds)
		else:
			fold_score = roc_auc_score(yte, fold_preds[:, 1])

		scores.append(fold_score)

	return scores