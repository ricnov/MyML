import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from Pipelines import pipelines
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import Configuration as C
import scipy, os, pickle
from scipy.stats import randint as sp_randint
import pandas as pd
from sklearn.model_selection import cross_validate
from FeatureEngineering import FeatureSelectionRFE, FeatureScaling, BoxCoxTransform,\
							YeoJohnsonTransform, LogTransform, PowerTransformationsWrapper, AutoML, CreatePolynomials, FeatureSelectionRFECV, PCAtransformer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB

__all__=["Hypertuning"]
__author__ = "Ruben Canadas Rodriguez"
__mail__ = "rubencr14@gmail.com"
__version__ = 1.0

#TODO: Apply parameter tuning for other methods also!

class ModelParameters(object):

	"""This class contains the different ranges of values that parameters can adopt"""

	def __init__(self, X, y, search):
		self._search = search
		self._features = len(X.columns.values)


	def extra_tree_classifier(self):
		
		params_grid = {  "n_estimators": range(10,100,10), 
					"criterion": ["gini", "entropy"], 
					"max_features": range(20, 100, 5),
					"min_samples_leaf": range(20,50,5),
					"min_samples_split": range(15,36,5)
					} 

		params_random = {"n_estimators": sp_randint(10,100), 
					"criterion": ["gini", "entropy"], 
					"max_features": sp_randint(10, self._features),
					"min_samples_leaf": sp_randint(20,50),
					"min_samples_split": sp_randint(15,45)
					} 					

		if self._search == "random": return params_random
		elif self._search == "grid": return params_grid

	def support_vector_machine_classifier(self):
		
		params_grid = { "kernel": ["linear", "sigmoid", "rbf"],
					"C":np.arange(1,10,0.1),
					"gamma": np.logspace(-4, 4, 100),
					"class_weight": ["balance", None]
		}
		
		params_random = { "kernel": ["linear", "sigmoid", "rbf"],
					"C":scipy.stats.uniform(1,3),
					"gamma": scipy.stats.uniform(0.1, 10),
					"class_weight": ["balanced", None],

		}
			
		if self._search == "random": return params_random
		elif self._search == "grid": return params_grid

	def voting_classifier(self):

		params_random = {"svc__kernel": ["linear", "sigmoid", "rbf"],
					"svc__C":scipy.stats.uniform(1,3),
					"svc__gamma": scipy.stats.uniform(0.1, 10),
					"svc__class_weight": ["balanced", None],
					"xt__n_estimators": sp_randint(10,100), 
					"xt__criterion": ["gini", "entropy"], 
					"xt__max_features": sp_randint(10, self._features),
					"xt__min_samples_leaf": sp_randint(20,50),
					"xt__min_samples_split": sp_randint(15,45)
		}
		
		return params_random



class Hypertuning(ModelParameters):

	"""This class inherits the parameters from its superclass ModelParameters in order to
	perform the grid/random search """

	def __init__(self, X, y, model, cv=10, verbose=True, search="random"):
		super(Hypertuning, self).__init__(X, y, search)
		self._model = model
		self._model_type = C.TYPE
		self._n_jobs = 20
		self.__cv = cv
		self.__verbose = verbose
		if self._model_type == "clf":
			self.__scoring = "accuracy"
		elif self._model_type == "reg":
			self.__scoring = "r2"
		if self._model =="extra_tree_classifier":
			self._params = self.extra_tree_classifier()
		elif self._model == "support_vector_machine_classifier":
			self._params = self.support_vector_machine_classifier()
		elif self._model == "voting":
			self._params = self.voting_classifier()
		else:
			raise ValueError("Method does not exist!")

	def parameter_search(self, X, y):


		if self._search == "grid":
			grid = GridSearchCV(param_grid=self._params, estimator=SVC(), scoring=self.__scoring, cv=self.__cv)
			grid.fit(X, y)
			if self.__verbose:
				print("Best parameters: {} with an score of {}".format(grid.best_params_, grid.best_score_))
			return grid_result.best_params_
		elif self._search == "random":
			grid = RandomizedSearchCV(param_distributions=self._params, estimator=VotingClassifier(estimators=[("svc", SVC(probability=True)), ("xt", ExtraTreesClassifier()), ("bn", BernoulliNB())], voting="soft", weights=[2,2,1]), 
										scoring=self.__scoring, cv=self.__cv, verbose=3, n_iter=7000, n_jobs=self._n_jobs)
			grid.fit(X, y)
			if self.__verbose:
				print("Best parameters: {} with an score of {}".format(grid.best_params_, grid.best_score_))
				df = pd.DataFrame().from_dict(grid.cv_results_)
				df.to_csv("grid_results.csv", sep=",")
			return grid.best_params_
