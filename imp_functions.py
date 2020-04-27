#importing the all most used libraries 
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################################

def basic_profile(data):
    '''
    data provided should be in the form of pandas dataframe
    
    provides back the basic steps for data exploration 
    '''
    import pandas as pd
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n\nBasic information of the dataset\n")
    print(data.info())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n\nGenerate descriptive statistics\n")
    print(data.describe())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n\nThe shape of the dataset is (rows x columns):",data.shape,'\n\n')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n\nThe types of column present in the dataset are :")
    print(data.get_dtype_counts(),'\n\n')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Columns with str data :",data.select_dtypes(include='object').columns)
    print("Columns with numerical data",data.select_dtypes(include='number').columns)
    print("Columns with categorical data",data.select_dtypes(include='category').columns)
    
    return print('\n\n\t\tEND')


###############################################################################################

def missing_val_per(data):
    """
    function to get the missing value from the dataset and also the percentage of the missing values 
    
    """
    total_cells = np.product(data.shape)
    total_missing_value = data.isnull().sum().sum()
    # percent of data that is missing

    percentage_missign_values_nfl = (total_missing_value / total_cells) * 100
    print("====================================")
    print(
        "Percentage of missing value in the dataset is :",
        ("%.2f" % round(percentage_missign_values_nfl, 2)),
        "%",
    )
    print("\n=====Missing Values per coloumn=====")
    return (data.isnull().sum(), "\n")


################################################################################################

def imbalance_check(target_class, data):
    '''
        target_class = target variable
        data = name of the dataset
        
        gives the visualization on whether the data is imbalanced or not 
        
        here 
    
    '''
    print('class 0 contributes to', round(data[target_class].value_counts()[0]/len(data) * 100,2), '% of the dataset')
    
    print('class 1 contributes to', round(data[target_class].value_counts()[1]/len(data) * 100,2), '% of the dataset')
    
    colors = ["#0101DF", "#DF0101"]
    plot = sns.countplot(x=target_class, data=data,palette=colors)
    return plot
    
###################################################################################################


def find_outliers(data):
    '''
    Check the outlier for a specific feature/column of the dataset
    
    '''
#     import numpy as np
    anomalies = []
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 3

    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    print(lower_limit)
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

###################################################################################################


def zscore_outliers(data):
    '''
    This function calculate the Z-score of the dataset and gives the outlier if present in the data
    
    ** the data in the dataframe should be numerical data
    '''

    from scipy import stats
    z = np.abs(stats.zscore(data))
    
    data_outlier = data[(z>3).all(axis=1)]
    return data_outlier.shape




