# from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(1)
print(y.shape,X.shape)
linear = MyLinearRegression()
# Create your k-fold splits or train-val-test splits as required
def partition(X,y,fold,k):
  splitted_X=np.array(np.array_split(X,k))
  splitted_y=np.array(np.array_split(y,k))
  testing_X=[]
  training_X=[]
  training_y=[]
  testing_y=[]
  for i in range(len(splitted_X)):
    if(i==fold):
      testing_X.extend(splitted_X[i])
      testing_y.extend(splitted_y[i])
    else:
      training_X.extend(splitted_X[i])
      training_y.extend(splitted_y[i])
  return np.array(training_X),np.array(testing_X),np.array(training_y),np.array(testing_y)


def validate(X,y,k,type):
  error=0.0
  min_err=float('inf')
  bestfold=0
  for fold in range(k):
    training_x,testing_x,training_y,testing_y=partition(X,y,fold,k)
    # print(training_x.shape,training_y.shape,testing_x.shape,testing_y.shape)
    theta=np.zeros((1,len(X.T)),dtype=float)
    linear.fit(training_x,training_y)
    # print(theta,bias)
    cost=linear.cost_function(testing_x,testing_y,type)
    if(min_err>cost):
      min_err=cost
      bestfold=fold
    error=error+cost
  return min_err,error/k,bestfold
min_err,error,bestfold=validate(X,y,10,1)
print(min_err,bestfold)
Xtrain,Xtest,ytrain,ytest = partition(X,y,bestfold,10)



linear.fit(X, y)
print(Xtest.shape)
ypred = linear.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)
def partition_logistic(X,y):
  splitted_X=np.array_split(X,10)
  splitted_y=np.array(np.array_split(y,10))
  # print(splitted_X.shape)
  test_x=[]
  train_x=[]
  val_x=[]
  test_y=[]
  train_y=[]
  val_y=[]
  for i in range(len(splitted_X)):
    if(i<7):
      train_x.extend(splitted_X[i])
      train_y.extend(splitted_y[i])
    elif(i==7):
      val_x.extend(splitted_X[i])
      val_y.extend(splitted_y[i])
    else:
      test_x.extend(splitted_X[i])
      test_y.extend(splitted_y[i])
  return np.array(train_x),np.array(test_x),np.array(val_x),np.array(train_y),np.array(test_y),np.array(val_y)
print('Logistic Regression')

X, y = preprocessor.pre_process(2)
print(y.shape,X.shape)
logistic = MyLogisticRegression()
Xtrain,Xtest,Xval,ytrain,ytest,yval=partition_logistic(X,y)
logistic.fit(Xtrain,ytrain)
ypred=logistic.predict(Xtest)
print('Predicted Values:', ypred)
print('True Values:', ytest)

