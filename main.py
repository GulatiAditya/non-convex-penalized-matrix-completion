import csv
import numpy as np
from matrix_completion import svt
from scipy.stats import bernoulli
from matplotlib import pyplot as plt

m = users = 943
n = items = 1682
data_file_name = "u.data" 
data = np.loadtxt(open(data_file_name,"rb"), delimiter = " ")

mask = np.zeros((m,n))
A = np.zeros((m,n))

for user, item, rating, _ in data:
	user = (int)(user)
	item = (int)(item)
	A[user-1][item-1] = rating
	# mask[user-1][item-1] = 1

mask = 1 - bernoulli.rvs(p=0.5, size=(m, n))
n_epochs = 100
l = [1000,500,100,50,5,1]
g = [100,20,5,1,0.5]
i = 0
for _lambda in l:
	for gamma in g:
		i += 1 
		solution, error4 = svt(A=A,mask=mask,regularisation='SCAD', _lambda = _lambda, gamma = gamma, epoch=n_epochs)
		plt.plot(range(len(error4)), error4, label='SCAD L:'+str(_lambda)+' g:'+str(gamma))
		plt.legend()
		plt.show()
		solution, error1 = svt(A=A,mask=mask,regularisation='MCP', _lambda = _lambda, gamma = gamma, epoch=n_epochs)
		plt.plot(range(len(error1)), error1, label='MCP L:'+str(_lambda)+' g:'+str(gamma))
		plt.legend()
		plt.show()
		solution, error2 = svt(A=A,mask=mask,regularisation='Soft', _lambda = _lambda, gamma = gamma, epoch=n_epochs)
		plt.plot(range(len(error2)), error2, label='Soft L:'+str(_lambda)+' g:'+str(gamma))
		plt.legend()
		plt.show()