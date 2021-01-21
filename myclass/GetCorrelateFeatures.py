import gzip
import pandas as pd
import re
import sys
import platform

from tqdm import tqdm

class GetCorrelateFeatures:
	def __init__(self, num_features=5):
		self.num_features = num_features
		return

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		best_features = list()
		mysample = X.copy()		

		mat = mysample.corr().abs() # calculate correlation matrix
		# old version
		with tqdm(total=100, file=sys.stdout) as bar:
			for i in range(0, self.num_features):
			  
				min_values = mat.min() # min of corr-value for each features, it's a Series 
				
				min_absolute = min_values.min() # min among all the minimum values
				min_col1 = min_values.idxmin() # select the index of the min 
				min_col2 = mat.idxmin()[min_col1]
				
				best_features.append(min_col1)
				best_features.append(min_col2)

				mat.drop([min_col1, min_col2], inplace=True, axis=1) # update the correlation matrix 
				bar.update(100/self.num_features)
		# del mysample
		X = X.loc[:, best_features]
		print(X.shape)
		return X