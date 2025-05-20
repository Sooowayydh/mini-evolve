"""
Matrix Multiplication Problem
Input: Two 10x10 matrices
Output: Their matrix product (10x10 matrix)
"""

def matmul(a, b):
    # <evolve>
    result = [[0 for _ in range(10)] for _ in range(10)]
    for i in range(10):
        for j in range(10):
            for k in range(10):
                result[i][j] += a[i][k] * b[k][j]
    return result
    # </evolve>

def helper_function():
    """This is a fixed helper function that won't be evolved."""
    return "I'm a helper!" 