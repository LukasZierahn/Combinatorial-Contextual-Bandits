import numpy as np

def matrix_to_tensor(matrix: np.ndarray, m: int , n: int) -> np.ndarray:
    matrix = matrix.reshape(n, m, n, m)
    return np.einsum("cadb->abcd", matrix)

def tensor_to_matrix(tensor: np.ndarray) -> np.ndarray:
    size = tensor.shape[0] * tensor.shape[-1]
    tensor = np.einsum("abcd->cadb", tensor)
    return tensor.reshape(size, size)