from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class Eigenfaces(object):

    def __init__(self):
        self.mean = None

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1. Center the face dataset by subtracting the average face.
        2. Perform Singular Value Decomposition using np.linalg.svd.

        Args:
            X: (N, D) numpy array where each row represents a flattened grayscale face image.
        Returns:
            U: (N, min(N, D)) numpy array of left singular vectors
            S: (min(N, D), ) numpy array of singular values
            Vt: (min(N, D), D) numpy array of transposed right singular vectors
        Hints:
            Consult the documentation for np.linalg.svd.
            Take particular care for the full_matrices argument and the shape of your output.
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        return U, S, Vt

    def compute_eigenfaces(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Use your svd function to compute the top k "eigenfaces."
        The eigenfaces of a dataset are the right singular vectors of the decomposition.

        Args:
            X: (N, D) numpy array where each row is a flattened face image.
            k: Number of eigenfaces to retain.
        Returns:
            Eigenfaces: (k, D) numpy array where each row represents an eigenface.
        Guiding Questions:
            What values of sigma represent eigenfaces that capture significant variance?
            Does np.linalg.svd return the principle components sorted by sigma for you?
        """
        _, _, Vt = self.svd(X)
        eigenfaces = Vt[:k, :]
        
        return eigenfaces

    def project(self, data: np.ndarray, eigenfaces: np.ndarray) -> np.ndarray:
        """
        Given a dataset and a collection of eigenfaces,
        project each image in the dataset onto the eigenfaces.

        Recall that the eigenfaces were built off of centered data.
        This has functional usage, since you can't be certain that
        your test data will have the same brightness as your train data.
        Thus, again, subtract the mean.

        While we won't check for looping,
        you may experience timeouts for slow solutions.

        Args:
            data: (M, D) numpy array where each row is a flattened face image.
            eigenfaces: (k, D) numpy array where each row represents an eigenface.
        Returns:
            projection: (M, k) the data represented as a linear combination of the eigenfaces
        """
        if self.mean is None:
            self.mean = np.mean(data, axis=0)
            
        data_centered = data - self.mean
        
        projection = data_centered @ eigenfaces.T
        
        return projection