from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FeatureGeneration(BaseEstimator, TransformerMixin):
	"""
	Feature Generation Ideas
	========================
	1. Sign aspect ratio counts.
	2. Whether detected camera is in front or rear position or not.
	"""
	
	def __init__(self):
		return None
	
	def fit(self, X, y):
		sign_aspect_ratio_count = X.groupby('SignAspectRatio')['SignAspectRatio']\
								   .transform(lambda x: len(x))
		X = X.assign(sign_aspect_ratio_count=sign_aspect_ratio_count)
		
		detected_camera_position = X['DetectedCamera'].isin([0, 1]).astype(np.int)
		X = X.assign(detected_camera_position=detected_camera_position)
		
		return self
	
	def transform(self, X, y=None):
		sign_aspect_ratio_count = X.groupby('SignAspectRatio')['SignAspectRatio']\
								   .transform(lambda x: len(x))
		X = X.assign(sign_aspect_ratio_count=sign_aspect_ratio_count)
		
		detected_camera_position = X['DetectedCamera'].isin([0, 1]).astype(np.int)
		X = X.assign(detected_camera_position=detected_camera_position)
		
		return X