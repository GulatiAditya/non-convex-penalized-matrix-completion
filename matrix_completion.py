from __future__ import division
import numpy as np
from thresholding import threshold, penalty



def projection(A, mask):
  return mask*A


def svt(A, mask, tau=None,delta=None, epsilon=1e-2, epoch=1000, regularisation='MCP',_lambda=30, gamma=20):
  e = []
  (m,n) = A.shape
  Y = np.zeros((m,n))
  if not tau:
    tau = 5 * np.sum(A.shape) / 2
  if not delta:
    delta = 1.2 * (m*n) / np.sum(mask)

  for i in range(epoch):

    U, S, V = np.linalg.svd(Y, full_matrices=False)

    S_t = threshold(S,regularisation, _lambda, gamma)

    X = np.linalg.multi_dot([U, np.diag(S_t), V])

    Y = Y + delta * projection( (A - X), mask)    

    error = np.linalg.norm(projection( (X - A),  mask))  / np.linalg.norm(projection(A, mask)) 
    obj = 0.5*np.linalg.norm(projection(X-A, mask))**2 + penalty(S_t, regularisation, _lambda, gamma)
    if(error<=1):
      e.append(error)
    # else:
    #   e.append(0.97)
    if i%10==0:
      print("Iteration: %i; Error: %.4f, Obj: %.4f" % (i + 1, error, obj))
    if error < epsilon:
      break
  return X, e


