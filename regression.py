from typing import List, Tuple

import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted values
            label: (N, 1) numpy array, the ground truth values
        Return:
            A float value denoting the error between real and predicted
        """
        pred_np = np.array(pred).flatten()
        true_np = np.array(label).flatten()
        
        sqerr = np.square(pred_np - true_np)
        mse = np.mean(sqerr)
        rmse = np.sqrt(mse)
        
        return rmse

    def construct_polynomial_feats(self, x: np.ndarray, degree: int) -> np.ndarray:
        """
        Given a feature matrix x, create a new feature matrix
        which is all powers of the features up to the provided degree.
        
        (Implementation is based on the docstring's specific output shapes)

        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                where N is the number of instances and D is the dimensionality of each instance
            degree: the max polynomial degree
        Return:
            feat:
                when x is 1-dimensional: (N, degree+1) numpy array
                when x is D-dimensional: (N, degree+1, D) numpy array
        """
        # Handle 1-dimensional case
        if x.ndim == 1:
            N = x.shape[0]
            # Initialize with ones for the bias term (x^0)
            feat = np.ones((N, degree + 1))
            # x_col needs to be (N, 1) for broadcasting powers
            x_col = x.reshape(-1, 1)
            # Create powers [x^1, x^2, ..., x^degree]
            powers = np.arange(1, degree + 1)
            feat[:, 1:] = x_col ** powers
            return feat
        
        # Handle D-dimensional case
        else:
            N, D = x.shape
            # Initialize with ones for the bias term (x^0)
            feat = np.ones((N, degree + 1, D))
            # Reshape x to (N, 1, D) for broadcasting
            x_expanded = x[:, np.newaxis, :]
            # Create powers [1, 2, ..., degree]
            powers = np.arange(1, degree + 1).reshape(1, -1, 1)
            # (N, 1, D) ** (1, degree, 1) -> (N, degree, D)
            feat[:, 1:, :] = x_expanded ** powers
            return feat


    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        # (N, 1+D) @ (1+D, 1) -> (N, 1)
        return xtest @ weight

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray) -> np.ndarray:
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you should use the numpy linear algebra function (np.linalg.pinv)
        """
        # weight = (X^T X)^-1 X^T y
        # Using pinv: weight = pinv(X) @ y
        X_pseudo_inv = np.linalg.pinv(xtrain)
        weight = X_pseudo_inv @ ytrain
        return weight

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using gradient descent.
        Initialize the weights with zeros.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        
        for _ in range(epochs):
            # 1. Predict
            y_pred = self.predict(xtrain, weight) # (N, 1)
            
            # 2. Calculate Gradient (based on L_linear, GD formula)
            # L = 1/(2N) * sum(y_pred - y_true)^2
            # dL/dw = 1/N * X^T * (y_pred - y_true)
            error = y_pred - ytrain # (N, 1)
            gradient = (1/N) * (xtrain.T @ error) # (D, N) @ (N, 1) -> (D, 1)
            
            # 3. Update weights
            weight = weight - (learning_rate * gradient)
            
            # 4. Calculate loss (RMSE) *after* update and record
            y_pred_new = self.predict(xtrain, weight)
            loss = self.rmse(y_pred_new, ytrain)
            loss_per_epoch.append(loss)
            
        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using stochastic gradient descent.
        Initialize the weights with zeros.
        Iterate sequentially as per NOTE.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []
        
        for _ in range(epochs):
            # Iterate sequentially (as per NOTE)
            for i in range(N):
                # Get the i-th sample
                x_i = xtrain[i:i+1, :] # Shape (1, D)
                y_i = ytrain[i:i+1, :] # Shape (1, 1)
                
                # 1. Predict for one sample
                y_pred_i = self.predict(x_i, weight) # (1, 1)
                
                # 2. Calculate Gradient (based on L_linear, SGD formula)
                # L = 1/2 * (y_pred - y_true)^2
                # dL/dw = X_i^T * (y_pred - y_true)
                error = y_pred_i - y_i # (1, 1)
                gradient = x_i.T @ error # (D, 1) @ (1, 1) -> (D, 1)
                
                # 3. Update weights
                weight = weight - (learning_rate * gradient)
                
                # 4. Calculate loss (RMSE) on *entire* dataset *after* update
                y_pred_all = self.predict(xtrain, weight)
                loss = self.rmse(y_pred_all, ytrain)
                loss_per_step.append(loss)
                
        return weight, loss_per_step

    def linear_fit_MBGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        batch_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a linear regression model using mini-batch gradient descent.
        Initialize the weights with zeros.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []
        
        for _ in range(epochs):
            # Iterate sequentially in batches
            indices = np.arange(N)
            # We don't shuffle, to match local tests
            
            for i in range(0, N, batch_size):
                batch_indices = indices[i : i + batch_size]
                if len(batch_indices) == 0:
                    continue
                    
                x_batch = xtrain[batch_indices]
                y_batch = ytrain[batch_indices]
                
                B = x_batch.shape[0] # Get actual batch size
                
                # 1. Predict for batch
                y_pred_batch = self.predict(x_batch, weight) # (B, 1)
                
                # 2. Calculate Gradient (based on L_linear, MBGD formula)
                # L = 1/(2B) * sum(y_pred - y_true)^2
                # dL/dw = 1/B * X_b^T * (y_pred_b - y_true_b)
                error = y_pred_batch - y_batch # (B, 1)
                gradient = (1/B) * (x_batch.T @ error) # (D, B) @ (B, 1) -> (D, 1)
                
                # 3. Update weights
                weight = weight - (learning_rate * gradient)
                
                # 4. Calculate loss (RMSE) on *entire* dataset *after* update
                y_pred_all = self.predict(xtrain, weight)
                loss = self.rmse(y_pred_all, ytrain)
                loss_per_step.append(loss)
                
        return weight, loss_per_step

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:
        """
        Fit a ridge regression model using the closed form solution
        """
        N, D = xtrain.shape
        
        # 1. Create Identity matrix
        I = np.identity(D)
        
        # 2. Adjust I to not regularize the bias term (index 0)
        I[0, 0] = 0
        
        # 3. Calculate (X^T X + lambda*I)
        term1 = xtrain.T @ xtrain + c_lambda * I
        
        # 4. Calculate X^T y
        term2 = xtrain.T @ ytrain
        
        # 5. Solve for weights
        # (X^T X + lambda*I) * weight = X^T y
        weight = np.linalg.solve(term1, term2)
        
        return weight

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-07,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using gradient descent.
        Initialize the weights with zeros.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        
        for _ in range(epochs):
            # 1. Predict
            y_pred = self.predict(xtrain, weight) # (N, 1)
            
            # 2. Calculate linear part of gradient
            error = y_pred - ytrain # (N, 1)
            linear_grad = (1/N) * (xtrain.T @ error) # (D, 1)
            
            # 3. Calculate regularization part of gradient
            # L_reg = (c_lambda / 2N) * w^T * w
            # dL_reg/dw = (c_lambda / N) * w
            reg_weight = np.copy(weight)
            reg_weight[0, 0] = 0 # Do not regularize bias
            reg_grad = (c_lambda / N) * reg_weight # (D, 1)
            
            # 4. Total gradient
            gradient = linear_grad + reg_grad
            
            # 5. Update weights
            weight = weight - (learning_rate * gradient)
            
            # 6. Calculate loss (RMSE) *after* update
            y_pred_new = self.predict(xtrain, weight)
            loss = self.rmse(y_pred_new, ytrain)
            loss_per_epoch.append(loss)
            
        return weight, loss_per_epoch

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using stochastic gradient descent.
        Initialize the weights with zeros.
        Iterate sequentially as per NOTE.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []
        
        for _ in range(epochs):
            for i in range(N):
                x_i = xtrain[i:i+1, :]
                y_i = ytrain[i:i+1, :]
                
                # 1. Predict
                y_pred_i = self.predict(x_i, weight)
                
                # 2. Calculate linear gradient
                error = y_pred_i - y_i
                linear_grad = x_i.T @ error # (D, 1)
                
                # 3. Calculate regularization gradient (as per formula L_ridge, SGD)
                # L_reg = (c_lambda / 2N) * w^T * w
                # dL_reg/dw = (c_lambda / N) * w
                reg_weight = np.copy(weight)
                reg_weight[0, 0] = 0 # Do not regularize bias
                reg_grad = (c_lambda / N) * reg_weight
                
                # 4. Total gradient
                gradient = linear_grad + reg_grad
                
                # 5. Update
                weight = weight - (learning_rate * gradient)
                
                # 6. Calculate loss on *entire* dataset
                y_pred_all = self.predict(xtrain, weight)
                loss = self.rmse(y_pred_all, ytrain)
                loss_per_step.append(loss)
                
        return weight, loss_per_step

    def ridge_fit_MBGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        batch_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a ridge regression model using mini-batch gradient descent.
        Initialize the weights with zeros.
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        loss_per_step = []
        
        indices = np.arange(N)
        
        for _ in range(epochs):
            # We don't shuffle, to match local tests
            for i in range(0, N, batch_size):
                batch_indices = indices[i : i + batch_size]
                if len(batch_indices) == 0:
                    continue

                x_batch = xtrain[batch_indices]
                y_batch = ytrain[batch_indices]
                
                B = x_batch.shape[0]
                
                # 1. Predict
                y_pred_batch = self.predict(x_batch, weight)
                
                # 2. Calculate linear gradient
                error = y_pred_batch - y_batch
                linear_grad = (1/B) * (x_batch.T @ error)
                
                # 3. Calculate regularization gradient (as per formula L_ridge, MBGD)
                # L_reg = (c_lambda / 2N) * w^T * w
                # dL_reg/dw = (c_lambda / N) * w
                reg_weight = np.copy(weight)
                reg_weight[0, 0] = 0 # Do not regularize bias
                reg_grad = (c_lambda / N) * reg_weight
                
                # 4. Total gradient
                gradient = linear_grad + reg_grad
                
                # 5. Update
                weight = weight - (learning_rate * gradient)
                
                # 6. Calculate loss on *entire* dataset
                y_pred_all = self.predict(xtrain, weight)
                loss = self.rmse(y_pred_all, ytrain)
                loss_per_step.append(loss)
                
        return weight, loss_per_step

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 5, c_lambda: float = 100
    ) -> List[float]:
        """
        For each of the k-folds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each fold
        """
        N = X.shape[0]
        indices = np.arange(N)
        # Use np.array_split for even folds, even if N % kfold != 0
        folds = np.array_split(indices, kfold)
        
        loss_per_fold = []
        
        for k in range(kfold):
            # 1. Get validation and training indices for this fold
            val_indices = folds[k]
            
            # Concatenate all other folds to create the training set
            train_indices_list = [folds[j] for j in range(kfold) if j != k]
            train_indices = np.concatenate(train_indices_list)
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            # 2. Fit model using closed-form ridge
            weight = self.ridge_fit_closed(X_train_fold, y_train_fold, c_lambda)
            
            # 3. Predict on the validation fold
            y_pred_val = self.predict(X_val_fold, weight)
            
            # 4. Calculate RMSE and store it
            loss = self.rmse(y_pred_val, y_val_fold)
            loss_per_fold.append(loss)
            
        return loss_per_fold

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []
        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm
        return best_lambda, best_error, error_list