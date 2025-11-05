from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def euclid_pairwise_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    You implemented this in project 2! We'll give it to you here to save you the copypaste.
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    """
    x_norm = np.sum(x**2, axis=1, keepdims=True)
    yt = y.T
    y_norm = np.sum(yt**2, axis=0, keepdims=True)
    dist2 = np.abs(x_norm + y_norm - 2.0 * (x @ yt))
    return np.sqrt(dist2)


def confusion_matrix_vis(conf_matrix: np.ndarray):
    """
    Fancy print of confusion matrix. Just encapsulating some code out of the notebook.
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix)
    ax.set_xlabel("Predicted Labels", fontsize=16)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Actual Labels", fontsize=16)
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(
            j,
            i,
            str(val),
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3"),
        )
    plt.show()
    return


class SMOTE(object):

    def __init__(self):
        pass

    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate the confusion matrix for the predicted labels of a classification task.
        This function should be able to process any number of unique labels, not just a binary task.

        The choice to put "true" and "predicted" on the left and top respectively is arbitrary.
        In other sources, you may see this format transposed.

        Args:
            y_true: (N,) array of true integer labels for the training points
            y_pred: (N,) array of predicted integer labels for the training points
            These vectors correspond along axis 0. y_pred[i] is the prediction for point i, whose true label is y_true[i].
            You can assume that the labels will be ints of the form [0, u).
        Return:
            conf_matrix: (u, u) array of ints containing instance counts, where u is the number of unique labels present
                conf_matrix[i,j] = the number of instances where a sample from the true class i was predicted to be in class j
        """
        u = np.max(y_true).astype(int) + 1
        
        conf_matrix = np.zeros((u, u), dtype=int)
        
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_label = y_pred[i]
            conf_matrix[true_label, pred_label] += 1
            
        return conf_matrix

    @staticmethod
    def interpolate(
        start: np.ndarray, end: np.ndarray, inter_coeff: float
    ) -> np.ndarray:
        """
        Return an interpolated point along the line segment between start and end.

        Hint:
            if inter_coeff==0.0, this should return start;
            if inter_coeff==1.0, this should return end;
            if inter_coeff==0.5, this should return the midpoint between them;
            to generalize this behavior, try writing this out in terms of vector addition and subtraction
        Args:
            start: (D,) float array containing the start point
            end: (D,) float array containing the end point
            inter_coeff: (float) in [0,1] determining how far along the line segment the synthetic point should lie
        Return:
            interpolated: (D,) float array containing the new synthetic point along the line segment
        """
        return start + inter_coeff * (end - start)

    @staticmethod
    def k_nearest_neighbors(points: np.ndarray, k: int) -> np.ndarray:
        """
        For each point, retrieve the indices of the k other points which are closest to that point.

        Hints:
            Find the pairwise distances using the provided function: euclid_pairwise_dist.
            For execution time, try to avoid looping over N, and use numpy vectorization to sort through the distances and find the relevant indices.
        Args:
            points: (N, D) float array of points
            k: (int) describing the number of neighbor indices to return
        Return:
            neighborhoods: (N, k) int array containing the indices of the nearest neighbors for each point
                neighborhoods[i, :] should be a k long 1darray containing the neighborhood of points[i]
                neighborhoods[i, 0] = j, such that points[j] is the closest point to points[i]
        """
        dists = euclid_pairwise_dist(points, points)
        
        np.fill_diagonal(dists, np.inf)
        
        sorted_indices = np.argsort(dists, axis=1)
        
        neighborhoods = sorted_indices[:, :k]
        
        return neighborhoods

    @staticmethod
    def smote(
        X: np.ndarray, y: np.ndarray, k: int, inter_coeff_range: Tuple[float]
    ) -> np.ndarray:
        """
        Perform SMOTE on the binary classification problem (X, y), generating synthetic minority points from the minority set.
        In 6.1, we did work for an arbitrary number of classes. Here, you can assume that our problem is binary, that y will only contain 0 or 1.

        Outline:
            # 1. Determine how many synthetic points are needed from which class label.
            # 2. Get the subset of the minority points.
            # 3. For each minority point, determine its neighborhoods. (call k_nearest_neighbors)
            # 4. Generate |maj|-|min| synthetic data points from that subset.
                # a. uniformly pick a random point as the start point
                # b. uniformly pick a random neighbor as the endpoint
                # c. uniformly pick a random interpolation coefficient from the provided range: `inter_coeff_range`
                # d. interpolate and add to output (call interpolate)
            # 5. Generate the class labels for these new points.
        Args:
            X: (|maj|+|min|, D) float array of points, containing both majority and minority points; corresponds index-wise to y
            y: (|maj|+|min|,) int array of class labels, such that y[i] is the class of X[i, :]
            k: (int) determines the size of the neighborhood around the sampled point from which to sample the second point
            inter_coeff_range: (a, b) determines the range from which to uniformly sample the interpolation coefficient
                Sample U[a, b)
                You can assume that 0 <= a < b <= 1
        Return:
            A tuple containing:
                - synthetic_X: (|maj|-|min|, D) float array of new, synthetic points
                - synthetic_y: (|maj|-|min|,) array of the labels of the new synthetic points
        """
        labels, counts = np.unique(y, return_counts=True)
        maj_label = labels[np.argmax(counts)]
        min_label = labels[np.argmin(counts)]
        num_maj = np.max(counts)
        num_min = np.min(counts)
        
        num_to_generate = num_maj - num_min
        
        X_minority = X[y == min_label]
        num_min_samples, D = X_minority.shape

        neighborhoods = SMOTE.k_nearest_neighbors(X_minority, k)
        
        synthetic_X = np.zeros((num_to_generate, D))
        
        for i in range(num_to_generate):
            rand_start_idx = np.random.randint(0, num_min_samples)
            start_point = X_minority[rand_start_idx]
            
            rand_neighbor_choice = np.random.randint(0, k)
            end_point_idx = neighborhoods[rand_start_idx, rand_neighbor_choice]
            end_point = X_minority[end_point_idx]
            
            a, b = inter_coeff_range
            inter_coeff = np.random.uniform(a, b)
            
            synthetic_X[i] = SMOTE.interpolate(start_point, end_point, inter_coeff)
            
        synthetic_y = np.full((num_to_generate,), min_label, dtype=int)
        
        return synthetic_X, synthetic_y


    @staticmethod
    def threshold_eval(y_true, y_pred, threshold) -> tuple[float, float]:
        """
        Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for a given (single) threshold.

        The threshold for the ROC curve is the value of the prediction at which we consider a point to be positive.
        For each threshold, we can calculate the true positive rate and false positive rate.

        Args:
            y_true: (N,) array or list of true integer labels
            y_pred: (N,) array or list of predicted float probabilities
            threshold: (float) in [0, 1] which determines the cutoff for classification

        Returns:
            A tuple containing the FPR and TPR for the given threshold.
            Mind the order of the return values (FPR, TPR).
        """
        y_pred_labels = (y_pred >= threshold).astype(int)
        
        TP = np.sum((y_pred_labels == 1) & (y_true == 1))
        FP = np.sum((y_pred_labels == 1) & (y_true == 0))
        TN = np.sum((y_pred_labels == 0) & (y_true == 0))
        FN = np.sum((y_pred_labels == 0) & (y_true == 1))
        
        tpr_denominator = TP + FN
        TPR = TP / tpr_denominator if tpr_denominator > 0 else 0.0
        
        fpr_denominator = FP + TN
        FPR = FP / fpr_denominator if fpr_denominator > 0 else 0.0
        
        return (FPR, TPR)

    @staticmethod
    def generate_roc(y_true, y_pred) -> list[Tuple[float, float]]:
        """
        Generate the sorted list of (FPR, TPR) points for the ROC Curve.

        The thresholds at which to evaluate the FPR and TPR are determined by the unique values in y_pred.
        This will generate precise ROC curve points, as the TPR and FPR will ony change at these values.

        In the threshold values ensure that you add, 0.0 and a value which is slightly greater that maximum probability
        to the unique values of y_pred.
        This will ensure that the edges cases of the ROC curve are covered.

        Since we want to model the Receiver Operating Characteristic (ROC) curve using the output of this function,
        FPR must be sorted in the ascending order. However, in the case of multiple points having the same FPR,
        we must further sort them by TPR in ascending order. This is important because the ROC curve should
        always move upwards when FPR remains constant.

        If the FPR values are not sorted correctly, the ROC will be misaligned and the curve will be having an incorrect shape
        which will lead to incorrect AUC value. Additionally, the corresponding TPR and threshold values must be aligned index-wise with
        their respective FPR values.

        This will ensure the correct plotting of the ROC curve and the accurate value of AUC.


        Args:
            y_true: (N,) array of true integer labels for the training points
            y_pred: (N,) array of predicted labels for the training points

        Returns:
            A list of (FPR, TPR) tuples, sorted by FPR and then TPR in ascending order to model the ROC curve.
        """
        thresholds = np.unique(y_pred)
        thresholds = np.unique(np.append(thresholds, [0.0, 1.01]))
        
        roc_points = []
        for t in thresholds:
            fpr, tpr = SMOTE.threshold_eval(y_true, y_pred, t)
            roc_points.append((fpr, tpr))
            
        roc_points.sort(key=lambda x: (x[0], x[1]))
        
        return roc_points

    @staticmethod
    def integrate_curve(roc_points: List[Tuple[float, float]]) -> float:
        """
        Calculate the Area Under the Curve (AUC) by integrating the ROC curve.

        Args:
            roc_points: A sorted list of (FPR, TPR) tuples, represetning the ROC curve as produced by the function 'generate_roc'.

        Returns:
            AUC: (float) The value of the Area Under the Curve (AUC) for the given ROC curve.
        """
        fpr_values = np.array([p[0] for p in roc_points])
        tpr_values = np.array([p[1] for p in roc_points])
        
        auc = np.trapz(tpr_values, fpr_values)
        
        return auc

    @staticmethod
    def plot_roc_auc(roc_points: List[Tuple[float, float]], auc: float):
        """
        Plot the ROC curve and display the AUC value of the curve.

        Args:
            roc_points A sorted list of (FPR, TPR) tuples, which represents the ROC AUC curve for the given data.
            auc: (float) The value of the Area Under the Curve (AUC) for the given ROC Curve.
        """
        fpr = np.array([p[0] for p in roc_points])
        tpr = np.array([p[1] for p in roc_points])
        fig, graph = plt.subplots(figsize=(8, 6))
        graph.plot(
            fpr,
            tpr,
            marker="o",
            linestyle="-",
            color="blue",
            label=f"ROC Curve (AUC = {auc:0.3f})",
        )
        graph.plot([0, 1], [0, 1], linestyle="--", color="red")
        graph.fill_between(fpr, tpr, color="blue", alpha=0.2)
        graph.set_title("Receiver Operating Characteristic (ROC) Curve")
        graph.set_xlabel("False Positive Rate (FPR)")
        graph.set_ylabel("True Positive Rate (TPR)")
        graph.legend()
        plt.show()