#MODIFICA DEL 7 NOV 2020 ALLA PENULTIMA RIGA
import pandas as pd
import numpy as np
import pickle
import platform

class Clean_Merge_Dataset:
	def __init__(self, name=''):
		
		return

	def fit(self, X, y=None):
		return self
		
	def transform(self, data_normal, data_tumor, X=None, y=None):
		
		dataset = pd.concat([data_tumor, data_normal], ignore_index=True)
		
		# removing TCGA-MESO label items
		dataset = dataset[dataset['label'] != 'TCGA-MESO']
		
		
		# union target with label
		for index, element in dataset.iterrows():
			if element['target'] is False:
				element['label'] = element['target']
			#else:
				# element['label'] = element['label'] + '_' + str(element['target'])
				#element['label'] = element['label']
			dataset.at[index, 'label'] = element['label']
		dataset.drop(['target'], inplace=True, axis=1)
		
		# Check null and zero value
		# count non zero value

		# delete before the 0-values features
		sum_count = 0
		index_to_delete = list()
		for i, element in enumerate(dataset.isin([0]).sum()):
			if element == dataset.shape[0]:
				sum_count += 1
				index_to_delete.append(i)
			
		dataset.drop(dataset.columns[index_to_delete], inplace=True, axis=1) # delete 0-values features
		

		# counting Nan values
		sum_count = 0
		index_to_delete = list()
		for i, element in enumerate(dataset.isna().sum()):
			if element == dataset.shape[0]:
				sum_count+=1
				index_to_delete.append(i)

		dataset.dropna(inplace=True, axis=1)
		
		
		
		del data_normal
		del data_tumor

		y = dataset.loc[:, 'label']
		#X = dataset.drop(columns=['label']) #ho cancellato 'case_id' 07-11-2020
		X = dataset
        #X = dataset.iloc[:, dataset.columns != 'case_id']
		
		return X, y
