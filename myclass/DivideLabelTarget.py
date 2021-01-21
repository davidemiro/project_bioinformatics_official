import gzip
import pandas as pd
import re
import os
import platform

class DivideLabel:
	def __init__(self):
		return 

	def fit(self, X, y):
		return

	def transform(self, X, y):
		y1 = pd.DataFrame(columns=['label'])
		y2 = pd.DataFrame(columns=['target'])

		for i, value in y.iterrows():
			label = value.split("_")[0]
			target = value.split("_")[1]
			y1 = y1.append()
			y2 = y2.append()
		return X, y1, y2