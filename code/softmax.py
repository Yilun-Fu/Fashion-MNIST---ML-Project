import numpy as np


import numpy as np
from sklearn.covariance import log_likelihood

DDBUG = 0


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_num: int):
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_num = batch_num

    def softmax2(self, X):
        # numericallly stable softmax
        max = np.max(X)
        Y = np.exp(X - max)
        sum = np.sum(Y)
        Y /= sum
        if DDBUG:
            print("Y", Y.shape, Y)
        return Y

    def softmax_2D(self, X):
        # numericallly stable version for 2D array
        max = np.max(X, axis=0, keepdims=1)
        Y = np.exp(X - max)
        sum = np.sum(Y, axis=0, keepdims=1)
        return Y / sum

    def cross_entropy(self, X, y):
        n = y.size
        return log_likelihood(X, y) / n     # loss(:float)

    def softmax(self, x_obs):
        """
        @Param  x_obs: a numpy array of shape (D,) containing one observation
        @Param  self.w: a numpy array of shape (V, D) containing V vectors of w_c(D,) 
        @return output: a numpy array of shape (V,) containing V values which sum to 1
        """
        D = len(x_obs)
        output = np.zeros((self.n_class,))
        # for c in range(self.n_class):
        #     output[c] = np.exp(self.w[c] @ x_obs)
        output = np.exp(self.w @ x_obs)
        output /= (np.sum(output))
        if DDBUG:
            print("w", self.w)
            print("output", output)
            print("output.sum", np.sum(output))
        return output

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        # X_train (N, D)
        # y_train (N, 1)
        # w       (V, D)
        # y_one_hot (N, V)

        N, D = X_train.shape
        y_onehot = np.zeros((N, self.n_class))
        # print(y_train)
        for i in range(N):
            idx = np.asscalar(y_train[i])
            y_onehot[i, idx] = 1

        f_c = self.softmax_2D(self.w @ X_train.T)       # f_c  shape(V, N)
        if DDBUG:
            print("f_c", f_c.shape)
        f_c -= y_onehot.T
        return f_c @ X_train                            # shape (V, D)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        N, D = X_train.shape

        self.w = np.zeros((self.n_class, D))
        X_train_batches = np.split(X_train, self.batch_num)
        y_train_batches = np.split(y_train, self.batch_num)
        n = X_train_batches[0].shape[0]

        for epoch in range(self.epochs):
            i = np.random.randint(len(X_train_batches))
            x_train_i = np.reshape(X_train_batches[i], (n, D))
            y_train_i = np.reshape(y_train_batches[i], (n, 1))
            grad = self.calc_gradient(x_train_i, y_train_i)
            self.w -= self.lr * grad + self.reg_const * self.w / N
            if DDBUG:
                print("============================", epoch,
                      i, "============================")
                print("grad:", grad)
                print("w", self.w)
            if epoch % 50 == 0:
                self.lr /= 2
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape

        y_hat = np.zeros((N, self.n_class))
        y_hat = X_test @ self.w.T
        pred = np.argmax(y_hat, axis=1)

        if DDBUG:
            print(self.w)
            print(y_hat)
            print(pred)

        return pred
