import numpy as np

def decompose_covariance_matrix(covariance_matrix):
	"""
	decompose the covariance matrix into its principal components
	"""
	U, s, Vt = np.linalg.svd(covariance_matrix, full_matrices=True)
	scaling = np.sqrt(s).tolist()
	rotation = np.arccos(U[0, 1])
	rotation_degree = rotation * (180 / np.pi)

	return scaling, rotation, rotation_degree