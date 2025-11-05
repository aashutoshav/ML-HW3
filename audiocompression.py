from typing import Tuple, Union
import numpy as np


class AudioCompression(object):

    def __init__(self):
        pass

    def svd(self, X: np.ndarray) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """        
        Perform SVD. You should use numpy's SVD.
        Your function should be able to handle single channel
        audio ((1, F, N) arrays) as well as stereo audio ((2, F, N) arrays)
        In this audio compression, we assume that each column of the spectrogram is a feature. Perform SVD on the channels of
        the audio (1 channel for single channel, 2 for stereo)
        The audio is the matrix X.
        
        Hint: np.linalg.svd by default returns the transpose of V. We want you to return the transpose of V, not V.
        
        Args:
            X: (C, F, N) numpy array corresponding to the audio
        
        Return:
            U: (C, F, F) numpy array
            S: (C, min(F, N)) numpy array
            V^T: (C, N, N) numpy array
        """
        U_list = []
        S_list = []
        V_list = []
        
        for i in range(X.shape[0]):
            U_c, S_c, V_c_T = np.linalg.svd(X[i], full_matrices=True)
            
            U_list.append(U_c)
            S_list.append(S_c)
            V_list.append(V_c_T)
            
        return np.stack(U_list), np.stack(S_list), np.stack(V_list)

    def compress(self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
        ) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """        
        Compress the SVD factorization by keeping only the first k components
        
        Args:
            U (np.ndarray): (C, F, F) numpy array
            S (np.ndarray): (C, min(F, N)) numpy array
            V (np.ndarray): (C, N, N) numpy array (This is V^T)
            k (int): int corresponding to number of components to keep
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                U_compressed: (C, F, k) numpy array
                S_compressed: (C, k) numpy array
                V_compressed: (C, k, N) numpy array
        """
        U_compressed = U[:, :, :k]
        
        S_compressed = S[:, :k]
        
        V_compressed = V[:, :k, :]
        
        return U_compressed, S_compressed, V_compressed

    def rebuild_svd(self, U_compressed: np.ndarray, S_compressed: np.
        ndarray, V_compressed: np.ndarray) ->np.ndarray:
        """        
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.
        
        Args:
            U_compressed: (C,F,k) numpy array
            S_compressed: (C,k) numpy array
            V_compressed: (C,k,N) numpy array (This is V_k^T)
        
        Return:
            Xrebuild: (C,F,N) numpy array
        
        Hint: numpy.matmul may be helpful for reconstructing stereo audio
        """
        U_scaled = U_compressed * S_compressed[:, np.newaxis, :]

        X_rebuild = U_scaled @ V_compressed
        
        return X_rebuild

    def compression_ratio(self, X: np.ndarray, k: int) ->float:
        """        
        Compute the compression ratio of a sample: (num stored values in compressed)/(num stored values in original)
        Refer to https://timbaumann.info/svd-image-compression-demo/ [cite: 682]
        Args:
            X: (C,F,N) numpy array
            k: int corresponding to number of components
        
        Return:
            compression_ratio: float of proportion of storage used by compressed audio
        """
        original_size = X.size
        
        C = X.shape[0]
        F = X.shape[1]
        N = X.shape[2]
        
        compressed_size = (C * F * k) + (C * k) + (C * k * N)
        
        return compressed_size / original_size

    def recovered_variance_proportion(self, S: np.ndarray, k: int) ->Union[
        float, np.ndarray]:
        """        
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation
        
        Args:
           S: (C, min(F,N)) numpy array
           k: int, rank of approximation
        
        Return:
           recovered_var: C floats corresponding to proportion of recovered variance for each channel
        """
        S_squared = S**2
        
        total_variance = np.sum(S_squared, axis=1)
        
        recovered_variance = np.sum(S_squared[:, :k], axis=1)
        
        total_variance[total_variance == 0] = 1e-9
        
        proportion = recovered_variance / total_variance
        
        if proportion.shape[0] == 1:
            return proportion[0]
        else:
            return proportion

    def memory_savings(self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V:
        np.ndarray, k: int) ->Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original audio X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (C,F,N) numpy array
            U (np.ndarray): (C,F,F) numpy array
            S (np.ndarray): (C,min(F,N)) numpy array
            V (np.ndarray): (C,N,N) numpy array
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """
        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (U_compressed.nbytes + S_compressed.nbytes +
            V_compressed.nbytes)
        savings = original_nbytes - compressed_nbytes
        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int=3) ->str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return '0B'
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f'{round(n, ndigits)} {units[units_idx]}'