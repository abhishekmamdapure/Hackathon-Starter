# Python Codes for Common Machine Learning Algorithms 

Following are the python codes format for most basic machine learning algorithms to save your time when performing a hackathon, just copy paste the format and use them to build your model. 


# Table of Contents
- [Linear Regression](#linear-regression) 
- [Logistic Regression](#logistic-regression)
- [Decision Tree](#decision-tree)
- [Random Forest](#random-forest)
- [Dimensionality Reduction Algorithms](#dimensionality-reduction-algorithms)
- [GBM](#gbm)
- [XG Boost](#xg-boost)
- [Cat Boost](#cat-boost)
- [Support Vector Machine](#support-vector-machine)
- [Naive Bayes](#naive-bayes)
- [kNN](#knn)
- [K Means](#k-means)



## Linear Regression 



```python
# importing required libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Build the model
model = LinearRegression()
# fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model
print('\nCoefficient of model :', model.coef_)
# intercept of the model
print('\nIntercept of model',model.intercept_)

# predict the target on the testing dataset
predict_test = model.predict(test_x)
print('\nItem_Outlet_Sales on test data',predict_test)
# Root Mean Squared Error on testing dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
```

## Logistic Regression 
```python 
# importing required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Build the model
model = LogisticRegression()
 # fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model
print('Coefficient of model :', model.coef_)
# intercept of the model
print('Intercept of model',model.intercept_)
# predict the target on the test dataset
predict_test = model.predict(test_x)
print('Target on test data',predict_test)

```


## Decision Tree

```python
# importing required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#build the decision tree model
model = DecisionTreeClassifier()
# fit the model with the training data
model.fit(train_x,train_y) 

# depth of the decision tree
print('Depth of the Decision Tree :', model.get_depth())
```


## Random Forest  
```python
# importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#build the model
model = RandomForestClassifier()

# fit the model with the training data
model.fit(train_x,train_y)

# number of trees used
print('Number of Trees used : ', model.n_estimators)
```

## Dimensionality Reduction Algorithms

* Use this to reduce the dimension of your dataset and then apply any of the prediction algorithm, mentioned above. 
```python
# importing required libraries
from sklearn.decomposition import PCA

model_pca = PCA(n_components=12) #build the PCA model

new_train = model_pca.fit_transform(train_x) 
new_test  = model_pca.fit_transform(test_x)

"""
new_train,new_test will have the less number of dimension 
than the original one
use the new created data (with less dimension) to build the model
""" 
 
```


## GBM

```python
from sklearn.ensemble import GradientBoostingClassifier

#building the model
model = GradientBoostingClassifier(n_estimators=100,max_depth=5)

# fit the model with the training data
model.fit(train_x,train_y)
```

## XG Boost

```python
#importig the library
import xgboost 
from xgboost import XGBClassifier

#building the model 
model = XGBClassifier()

# fit the model with the training data
model.fit(train_x,train_y)
```

## Cat Boost

```python
#importing the required library
from catboost import CatBoostRegressor
#building the model 
model = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE') 
#fit the model with the training data 
model.fit(X,y)
```

## Support Vector Machine

```python
# importing the required library
from sklearn.svm import SVC

# building the model
model = SVC()

# fit the model with the training data
model.fit(train_x,train_y)
```

## Naive Bayes

```python 
# importing the required library
from sklearn.naive_bayes import GaussianNB

# building the model
model = GaussianNB()

# fit the model with the training data
model.fit(train_x,train_y)
```
## kNN 
```python
#importing the required library
from sklearn.neighbors import KNeighborsClassifier

# building the model 
model = KNeighborsClassifier()  

# fit the model with the training data
model.fit(train_x,train_y)
```

## K-Means

```python
# importing required libraries
from sklearn.cluster import KMeans

# building the model
model = KMeans()  

# fit the model with the training data
model.fit(train_data)
```





