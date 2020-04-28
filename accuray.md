# Python Codes for performance metrics of ML algorithms

Following are the python codes format for performance metrics of machine learning algorithms to save your time when performing a hackathon, just copy paste the format and use them to build your model. 



# Performance metrics for Classification problems

> Let us define sample actual and predicted values to understand the working of performance metrics for classification problem.
```python 
X_actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
Y_pred = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0]
```  

- [Confusion Matrix](#confusion-matrix) 
- [Accuracy Score](#accuracy-score)
- [Classification Report](#classification-report)
- [ROC AUC Score](#roc-auc-score)
- [Log Loss](#log-loss)

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
results = confusion_matrix(X_actual, Y_predic)
print ('Confusion Matrix :')
print(results)
```
Output:
```python
Confusion Matrix :
[[3 3]
 [1 3]]
```

### Accuracy Score
```python
from sklearn.metrics import accuracy_score
print ('Accuracy Score is',accuracy_score(X_actual, Y_pred))
```
Output:
```python
Accuracy Score is 0.6
```

### Classification Report
```python
from sklearn.metrics import classification_report
print ('Classification Report : ')
print (classification_report(X_actual, Y_pred))
```
Output:
```python
Classification Report : 
              precision    recall  f1-score   support

           0       0.75      0.50      0.60         6
           1       0.50      0.75      0.60         4

    accuracy                           0.60        10
   macro avg       0.62      0.62      0.60        10
weighted avg       0.65      0.60      0.60        10
```

### ROC AUC Score
```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
print('AUC-ROC:',roc_auc_score(X_actual, Y_pred))
```
Output:
```python
AUC-ROC: 0.625
```

### Log Loss
```python
from sklearn.metrics import log_loss
print('LOGLOSS Value is',log_loss(X_actual, Y_pred))
```
Output:
```python
LOGLOSS Value is 13.815750437193334
```

# Performace metrics for Regression problems

> Let us define sample actual and predicted values to understand the working of performance metrics for regression problem. 
```python
X_actual = [5, -1, 2, 10]
Y_pred = [3.5, -0.9, 2, 9.9]
```

- [R Squared](#r-squared)
- [Mean Absolute Error](#mean-absolute-error)
- [Mean Squared Error](#mean-squared-error)


### R Squared
```python
from sklearn.metrics import r2_score
print ('R Squared =',r2_score(X_actual, Y_pred))
```
Output:
```python
R Squared = 0.9656060606060606
```

### Mean Absolute Error

```python
from sklearn.metrics import mean_absolute_error
print ('MAE =',mean_absolute_error(X_actual, Y_pred))
```
Output:
```python
MAE = 0.42499999999999993
```

### Mean Squared Error

```python
from sklearn.metrics import mean_squared_error
print ('MSE =',mean_squared_error(X_actual, Y_pred))
```
Output:
```python
MSE = 0.5674999999999999
```
