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

def ejercicio_1(c):
	# Load the dataset
	data = pd.read_csv('datasetsLA4IMC/BasesDatos/csv/dataset2.csv', header=None)
	X = data.iloc[:,:-1].values
	Y = data.iloc[:,-1].values
	
	# Train the SVM model
	svm_model = svm.SVC(kernel='linear', C=c)
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
	
	# Make a color plot including the margin hyperplanes (Z=-1) and Z=separating hyperplane (Z=0)
	Z = Z.reshape(XX.shape)
	plt.pcolormesh(XX, YY, Z > 0)
	plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--'], levels=[-1,0,1])
	plt.show()
	

def ejercicio_2(c, g):
	# Load the dataset
	data = pd.read_csv('datasetsLA4IMC/BasesDatos/csv/dataset2.csv', header=None)
	X = data.iloc[:,:-1].values
	Y = data.iloc[:,-1].values
	
	# Train the SVM model
	svm_model = svm.SVC(kernel='rbf', C=c, gamma=g)
	svm_model.fit(X, Y)
	
	# Show the points
	plt.figure(1)
	plt.clf()
	plt.scatter(X[:,0], X[:,1], c=Y, zorder=10, cmap=plt.cm.Paired)
	plt.show()
	
	
def main():
	print("SEGUNDO DATASET")
	
	ejercicio_2(1e4, 2e-1)
	
	
	print("Ejercicio 1")
	C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000]
	for i in C_range:
		print('Parametro C: ', i)
		ejercicio_1(i)
		
	print("Ejercicio 2 - SVM no lineal con kernel tipo RBF")
	g_range = [2e-2, 2e-1, 2e0, 2e1, 2e2]
	for i in g_range:
		print("C: ", 100, "; g: ", i)
		ejercicio_2(100,i)
		
	print("Ejercicio 2 - Combinaciones de C y gamma")
	for i in C_range:
		for j in g_range:
			print("Parametro C: ", i, "; g: ", j)
			ejercicio_2(i,j)
			
	print("Ejercicio 2 - Combinaciones de C y gamma")
	for i in g_range:
		for j in g_range:
			print("Parametro C: ", i, "; g: ", j)
			ejercicio_2(i,j)
			
	ejercicio_2(1e4, 2e-1)
	
	
if __name__ == '__main__':
	main()
