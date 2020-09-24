import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from scipy import stats

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            data1=pd.read_csv("/content/drive/My Drive/ML DataSet/abalone.txt",header=None)
            data1=data1.rename(columns={0:"Sex",1:"Length",2:"Diameter",3:"Height",4:"Whole Weight",5:"Shucked Weight",6:"Viscera Weight",7:"Shell Weight",8:"Rings"})
            X=data1.iloc[:,1:8].to_numpy()
            y=data1.iloc[:,8:].to_numpy()
            y=y+1.5
            pass
        elif dataset == 1:
            data1=pd.read_csv("/content/drive/My Drive/ML DataSet/Dataset1_A1")
            data1=data1[data1['Critic_Score'].notna()]
            data1=data1[data1['User_Score'].notna()]
            data1['User_Score']=pd.to_numeric(data1['User_Score'],errors='coerce')
            data1=data1[data1['User_Score'].notna()]
            data1=data1[['Critic_Score','User_Score','Global_Sales']]
            z=np.abs(stats.zscore(data1))
            data1=data1[(z<3).all(axis=1)]
            data1=pd.DataFrame(Normalizer().fit_transform(data1))
            X=data1.iloc[:,:2].values
            y=data1.iloc[:,2:].values
            y=y.reshape(-1,1)
            # Implement for the video game dataset
            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            df=pd.read_csv('/content/drive/My Drive/ML DataSet/data_banknote_authentication.txt',sep=',',header=None)
            df=df.rename(columns={0:"Variance", 1:"Skewness", 2:"Curtosis", 3:"Entropy",4:"Class"})
            X=np.array(df.iloc[:,:4])
            y=np.array(df.iloc[:,4:])
            pass

        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """
    def __init__(self):
        self.theta=np.empty((0))
        self.bias=0.0
        self.cost=np.empty((0))
        pass
    def update_thetas(self,y, X,type,learning_rate):
        """type: indicates the type of error 1: RMSE otherwise MAE"""
        ngames=len(y)
        if(type==1):
            inner=np.subtract(np.dot(self.theta,X.T),y.T)+self.bias 
            cost=2*self.cost_function(X,y,type)
            theta_deriv=(np.dot(inner,X))
            theta_deriv=theta_deriv/(ngames*cost)
            bias_deriv=inner.sum()/(ngames*cost)
            self.theta=self.theta-theta_deriv*learning_rate
            self.bias=self.bias-(bias_deriv*learning_rate)
        else:
            inner=np.subtract(self.theta@X.T,y.T)+self.bias
            inner=inner/np.abs(inner)
            theta_deriv=(inner@X)
            theta_deriv=theta_deriv/ngames
            bias_deriv=inner.sum()/ngames
            self.theta=self.theta-theta_deriv*learning_rate
            self.bias=self.bias-(bias_deriv*learning_rate)
        return self
    def cost_function(self,X,y,type):
        """type: indicates the type of error 1: RMSE otherwise MAE"""
        ngames=len(y)
        if(type==1):
            total_error=np.sum(((self.theta@X.T+self.bias)-y.T)**2)
            return math.sqrt(total_error/ngames)
        else:
            total_error=np.sum(np.abs((self.theta@X.T+self.bias)-y.T))
            return total_error/ngames
    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        learning_rate=0.001
        iters=1000
        type=0
        self.cost=np.zeros((iters,1))
        self.theta=np.zeros((1,len(X.T)))
        for i in range(iters):
            self.update_thetas(y,X,type,learning_rate)
            loss=self.cost_function(X,y,type)
            self.cost[i]=loss

        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        return X@self.theta.T + self.bias


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        self.theta=np.empty((0))
        self.bias=0.0
        self.cost=np.zeros((0))
    def sigmoid(self,X):
        """calculates sigmoid of an np array"""
        return (1/(1+np.exp(-X)))

    def cost_function(self,X,y):
        """ Evaluates the cost function"""
        htheta=self.sigmoid(self.theta@X.T+self.bias)  #(1 x m)
        cost=np.add((np.log(htheta))@y,(np.log(1-htheta))@(1-y))
        cost=-1*cost/len(y)
        return cost.flatten()[0]
    def update_thetas(self,X,y,learning_rate):
        """Evaluates the process """
        htheta=self.sigmoid((self.theta@X.T)+self.bias)  #(1 x m)
        theta_deriv=((htheta-y.T)@X)/len(y)
        self.theta=self.theta-(learning_rate*theta_deriv)
        self.bias=self.bias-learning_rate*float(np.sum(htheta-y.T,axis=1))
    def stochastic_gradient_descent(self,X,y,learning_rate,iters):
      self.cost=np.zeros((iters,1))
      self.theta=np.zeros((1,len(X.T)))
      n=len(y)
      X_old=X.copy()
      y_old=y.copy()
      for i in range(iters):
        X=X_old.copy()
        y=y_old.copy()
        p=np.random.permutation(n)
        index=np.random.randint(1,n)
        X=X[p]
        y=y[p]
        X=X[:index,:]
        y=y[:index,:]
        htheta=self.sigmoid((self.theta@X.T)+self.bias)
        loss=self.cost_function(X,y)
        self.cost[i]=loss
        self.theta=self.theta-learning_rate*((htheta-y.T)@X)
        self.bias=self.bias-learning_rate*((htheta-y.T).sum())
    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        self.theta=np.zeros((1,len(X.T)))
        learning_rate=0.1
        iters=80000
        self.cost=np.zeros((iters,1))
        for i in range(iters):
            self.update_thetas(X,y,learning_rate)
            loss=self.cost_function(X,y)
            self.cost[i]=loss
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        return np.round(self.sigmoid((X@self.theta.T)+self.bias))