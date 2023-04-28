#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:14:36 2022

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm

def ejercicio_1():
	# Load the dataset
	data = pd.read_csv('datasetsLA4IMC/BasesDatos/csv/dataset3.csv', header=None)
	X = data.iloc[:,:-1].values
	Y = data.iloc[:,-1].values
	
	# Train the SVM model
	svm_model = svm.SVC(kernel='linear', C=100)
	svm_model.fit(X, Y)
	
	# Show the points
	plt.figure(1)
	plt.clf()
	plt.scatter(X[:,0], X[:,1], c=Y, zorder=10, cmap=plt.cm.Paired)
	plt.show()
	

def ejercicio_2(c, g):
	# Load the dataset
	data = pd.read_csv('datasetsLA4IMC/BasesDatos/csv/dataset3.csv', header=None)
	X = data.iloc[:,:-1].values
	Y = data.iloc[:,-1].values
	
	# Train the SVM model
	svm_model = svm.SVC(kernel='rbf', C=c, gamma=g)
	svm_model.fit(X, Y)
	
	# Show the points
	plt.figure(1)
	plt.clf()
	plt.scatter(X[:,0], X[:,1], c=Y, zorder=10, cmap=plt.cm.Paired)
	
	# Show the separating hyperplane
	plt.axis('tight')
	
	# Extract the limit of the data to construct the mesh
	x_min = X[:,0].min()
	x_max = X[:,0].max()
	y_min = X[:,1].min()
	y_max = X[:,1].max()
	
	# Create the mesh and obtain the Z value returned by the SVM
	XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
	Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
	
	# Make a color plot including the margin hyperplanes (Z=-1 and Z=1) and the separating hyperplane (Z=0)
	Z = Z.reshape(XX.shape)
	plt.pcolormesh(XX, YY, Z > 0)
	plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1,0,1])
	plt.show()
	
	
def main():
	print("TERCER DATASET")
	ejercicio_1()
	print("SVM no lineal para clasificar datos - Valores optimos")
	ejercicio_2(1e4, 2e-1)
	print("Configuracion infra-entrenamiento")
	ejercicio_2(1, 2e-1)
	print("Configuracion sobre-entrenamiento")
	ejercicio_2(10, 2e2)
	
	
# CREAR MAIN
if __name__ == '__main__':
	main()
