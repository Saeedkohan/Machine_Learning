import math
import numpy as np

def det(A):
  if A.shape == (2, 2):
     return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

  deter = 0
  for i in range(A.shape[1]):

      m = np.delete(np.delete(A, 0, axis=0), i, axis=1)
      deter += ((-1) ** i) * A[0, i] * det(m)

  return deter


def normalize(vector):   
    norm = math.sqrt(sum(map(lambda vector:vector**2, vector)))
    return list(map(lambda v:v/norm, vector))

def eigen(matrix):
    b = -(matrix[0,0] + matrix[1,1])
    c = det(matrix)
    d = b**2 - 4  * c
 
    if d < 0:
        print("no roots")
    else:
        sqrt = math.sqrt(d)
        eigenvalue1 = (-b + sqrt) / 2 
        eigenvalue2 = (-b - sqrt) / 2 
 
    eigenvector1 = [eigenvalue1 - matrix[0][0], matrix[1][0]]
    eigenvector2 = [eigenvalue2 - matrix[1][1], matrix[1][0]]
        
    # if matrix[0][1] != 0:
    #     print("here")
    #     print(matrix[1][1])
 
    #     print(eigenvector1,eigenvector2)
    # else:
    #     print("here2")

    #     eigenvector1 = [1, 0]
    #     eigenvector2 = [0, 1]

    eigenvector1 = normalize(eigenvector1)
    eigenvector2 = normalize(eigenvector2)
    
    return [eigenvalue1, eigenvalue2], [eigenvector1, eigenvector2]

def svd(A):
    AtA =A.T@A
    eigenvalues, eigenvectors = eigen(AtA)
    
    Sigma = np.array([[math.sqrt(abs(eigenvalues[0])), 0], [0, math.sqrt(abs(eigenvalues[1]))]])
    
    V = np.array([eigenvectors[0], eigenvectors[1]]) 
    U =  A@V@ Sigma.T

A =np .array ([[4, 0], [3, -5]])

svd(A)
