import numpy as np


def ny_uppgift(name):
    print("\n\n============== Uppgift "+name+" ==============\n")


def gauss_elimination(A, b):
    print("Before gaussian elimination")
    print(A)
    print(b)
    eps = 0.001
    n = A.shape[0]
    for j in range(n-1):
        if abs(A[j, j]) < eps:
            raise ValueError("zero pivot encountered;")
        for i in range(j+1, n):
            mult = A[i, j] / A[j, j]
            for k in range(j, n): # k blir kolonn-index och måste börja på j
                A[i, k] -= mult * A[j, k]
            b[i] -= mult * b[j]
    print("After gaussian elimination:")
    print(A)
    print(b)

def back_substitution(A, b):
    n = len(A)
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            b[i] -= A[i, j] * x[j]
        x[i] = b[i] / A[i, i]
    return x

ny_uppgift("1a")
R1, R2, R3 = 3.0, 2.0, 8.0
U1, U2 = 10.0, 0.6

A = np.array([[1.0, -1.0, -1.0], [R1, R2, 0.0], [0.0, R2, -R3]])
b = np.array([[0.0], [U1], [U2]])
gauss_elimination(A, b)
x = back_substitution(A, b)
print("The result is:")
print(x)

ny_uppgift("1c")
A = np.random.rand(3,3)
b = np.random.rand(3,1)
AA, bb = A.copy(), b.copy()
gauss_elimination(A, b)
x = back_substitution(A, b)
print("The result is:")
print(x)
print("Compared with linalg.solve():")
print(np.linalg.solve(AA, bb))

ny_uppgift("2a")
