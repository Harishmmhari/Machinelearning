import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_featuers=X.shape
        self.weights=np.zeros(n_featuers)
        self.bias=0
        for _ in range(self.n_iters):
            linear_model=np.dot(X,self.weights)+self.bias
            y_pred=self._sigmoid(linear_model)
            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db


    def predict(self,X):
        linear_model=np.dot(X,self.weights)+self.bias
        y_pred=self._sigmoid(linear_model)
        y_predclas=[round(i) for i in y_pred]
        return  y_predclas

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))