import scipy as sp
import numpy as np
from cross_validation import cv_loop

SEED = 12313

def greedy_feature_search(Xts, y, model):
	score_hist = []
	N = 10
	good_features = set([])

	# Greedy feature selection loop
	while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
		scores = []
		for f in range(Xts.shape[1]):
			if f not in good_features:
				feats = list(good_features) + [f]
				Xt = Xts.iloc[:, feats]
				score = cv_loop(Xt, y, model, 'auc', SEED)
				scores.append((np.mean(score), f))
				print("Feature: %i Mean AUC: %f" % (f, np.mean(score)))
		good_features.add(sorted(scores)[-1][1])
		score_hist.append(sorted(scores)[-1])
		print("Current features: %s" % sorted(list(good_features)))
	
	# Remove last added feature from good_features
	good_features.remove(score_hist[-1][1])
	good_features = sorted(list(good_features))
	print("Selected features %s" % good_features)