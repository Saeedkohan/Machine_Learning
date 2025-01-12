# import numpy as np

# def  gauss_jordan_elimination(matrix):

#     n = len(matrix)
#     m = len(matrix[0])

#     for i in range(n):
#         max_row = i + np.argmax(np.abs(matrix[i:, i]))
#         matrix[[i, max_row]] = matrix[[max_row, i]]

#         print(matrix)
#         print("//")
#         pivot = matrix[i][i]
#         if pivot == 0:
#             continue
#         matrix[i] = matrix[i] / pivot
#         for j in range(n):
#             if j != i:
#                 matrix[j] -= matrix[i] * matrix[j][i]

#     return matrix

# matrix = np.array([[-1,2,0],[2,-4,0],[0,0,-9]], dtype=float)
# result = gauss_jordan_elimination(matrix)
# d=np.linalg.solve(result,[1,1,1])
# # print(result)

# print(d)


# تعریف ماتریس
A = [[8, 2, 0],
     [2, 5, 0],
     [0, 0, 0]]


eigenvalues = [0, 9, 4]


def subtract_lambda_I(matrix, eigenvalue):
    identity_matrix = [[1 if i == j else 0 for j in range(len(matrix))] for i in range(len(matrix))]
    result = [[matrix[i][j] - eigenvalue * identity_matrix[i][j] for j in range(len(matrix))] for i in range(len(matrix))]
    return result


def solve_for_eigenvector(matrix):
  
    if matrix[0][0] != 0:
        return [1, -matrix[0][1] / matrix[0][0], 0]
    elif matrix[1][1] != 0:
        return [0, 1, -matrix[1][2] / matrix[1][1]]
    else:
        return [0, 0, 1]


for eigenvalue in eigenvalues:
    matrix_lambda_I = subtract_lambda_I(A, eigenvalue)
    print(matrix_lambda_I)
    eigenvector = solve_for_eigenvector(matrix_lambda_I)
    print(f"Eigenvector for eigenvalue {eigenvalue}: {eigenvector}")

print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")

