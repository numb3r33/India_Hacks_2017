"""
Feature Library
----------------

This library contains list of different types of features used in different models.
"""

import numpy as np

from util import get_feature_map
from util import remove_colons
from util import remove_commas
from util import count_feature_instances

from natsort import natsorted



def create_feature_with_low_card(feature, feature_name):
	"""
	Clusters feature values based on feature mapping created
	Assuming that for every feature we would have to run remove colons
	"""

	def create_cluster(feature):
		feature_map = get_feature_map(feature_name)

		for replacement_key in feature_map.keys():
			to_replace = feature_map[replacement_key]
			feature    = feature.str.replace(r'%s'%(replacement_key), to_replace)

		feature = feature.map(lambda x: ''.join(list(natsorted(set(x)))))

		return feature


	feature = remove_colons(feature)
	feature = remove_commas(feature)

	return create_cluster(feature)


def calculate_count_features(data, features):
	count_features = []

	for feature in features:
		count_features.append(count_feature_instances(data.loc[:, feature]))

	count_features = np.array(count_features)
	return count_features.T

def calculate_watch_time(genres):
	return genres.str.replace(r'.*:', '').map(lambda x: np.sum(list(map(int, x.split(',')))))
