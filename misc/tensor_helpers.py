import numpy as np

def matrix_to_tensor(matrix: np.ndarray, m: int , n: int) -> np.ndarray:
    matrix = matrix.reshape(m, n, m, n)
    return np.einsum("acbd->abcd", matrix)

def tensor_to_matrix(tensor: np.ndarray) -> np.ndarray:
    size = tensor.shape[0] * tensor.shape[-1]
    tensor = np.einsum("abcd->acbd", tensor)
    return tensor.reshape(size, size)