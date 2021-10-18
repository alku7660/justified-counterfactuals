"""
Support functions & imports
"""

import numpy as np
import math
from scipy.spatial import distance_matrix
from itertools import permutations
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
from scipy.stats import rankdata
from sklearn import svm
import os
path = os.path.abspath('')
dataset_dir = str(path)+'/Datasets/'

def verify_diff_label(label,model,v):
    """
    Function that verifies if the label given does not match that of the model predicted on instance v
    Input label: Label to be compared
    Input model: Model to be used on instance v
    Input v: Instance to be checked on same label
    Output different: Boolean indicating whether labels are the same or not
    """
    different = False
    v = v.reshape(1, -1)
    label_v = model.predict(v)
    if label_v != label:
        different = True
    return different

def verify_same_label(label,model,v):
    """
    Function that verifies if the label given matches that of the model predicted on instance v
    Input label: Label to be compared
    Input model: Model to be used on instance v
    Input v: Instance to be checked on same label
    Output same: Boolean indicating whether labels are the same or not
    """
    same = True
    v = v.reshape(1, -1)
    label_v = model.predict(v)
    if label_v != label:
        same = False
    return same

def permutation_verify(x,vector,perm,label,model):
    """
    Auxiliary method that verifies a single permutation for binary features
    Input x: Instance of interest
    Input vector: vector of movement between x and target instance
    Input perm: Permutation to test
    Input model: Prediction model used
    Input label: Label of the instance of interest
    Output fail: Whether the permutation could not be verified in terms of the label
    """
    fail = 0
    v = np.copy(x)
    if type(perm) == list:
        for j in perm:
            v[j] += vector[j]
            if verify_diff_label(label,model,v):
                fail = 1
                break
    else:
        v[perm] += vector[perm]
        if verify_diff_label(label,model,v):
            fail = 1
    return fail

def euclidean(x1,x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def sort_data_distance(x,data,data_label):
    """
    Function to organize dataset with respect to distance to instance x
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input data_label: Training dataset label
    Input data_str: Dataset name
    Output data_sorted_distance: Training dataset sorted by distance to the instance of interest x
    """
    sort_data_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

def nn_to_cf(cf,train,train_target,x_label,data_str):
    """
    Method that finds the closest training instance to the counterfactual of interest
    """
    data_distance = sort_data_distance(cf,train,train_target)
    for i in data_distance:
        if i[2] != x_label:
            nn_cf = i[0]
            label_nn_cf = i[2]
            break
    return nn_cf, label_nn_cf

def corr_matrix(data_kNN,feat_type):
    """
    Function that calculates the correlation matrix for kNN around an instance of interest
    Input data_kNN: kNN instances around the instance x of interest
    Input feat_type: Type of the features used
    Output corr_matrix: Correlation matrix calculated from the features in the kNN
    """
    co_matrix = np.zeros((len(data_kNN[0]),len(data_kNN[0])))
    for i in range(len(data_kNN[0])):
        for j in range(i,len(data_kNN[0])):
            if i == j:
                corr_ij = 1
            else:
                feature_i = data_kNN[:,i]
                feature_j = data_kNN[:,j]
                feature_i_type = feat_type[i]
                feature_j_type = feat_type[j]
                if len(np.unique(feature_i)) == 1 or len(np.unique(feature_j)) == 1:
                    corr_ij = 0.0    
                else:
                    corr_ij = corr_value(feature_i,feature_j,feature_i_type,feature_j_type)
            co_matrix[i,j] = corr_ij
            co_matrix[j,i] = corr_ij
    return co_matrix

def corr_value(x,y,x_type,y_type):
    """
    Function that returns the correlation coefficient according to the type of the features
    Input x: Feature x
    Input y: Feature y
    Input x_type: Type of feature x
    Input y_type: Type of feature y
    Output corr_val: Correlation coefficient for the pair of variables with the right method of calculation
    """
    if x_type == 'cont' and y_type == 'cont':
        corr_val, p_val = spearmanr(x,y)
    elif x_type == 'cont' and y_type == 'bin':
        if np.isin(y[1:], [0,1]).all():
            corr_val = rank_biserial(x,y)
        else:
            corr_val, p_val = spearmanr(x,y)
    elif x_type == 'bin' and y_type == 'cont':
        if np.isin(x[1:], [0,1]).all():
            corr_val = rank_biserial(y,x)
        else:
            corr_val, p_val = spearmanr(x,y)
    elif x_type == 'bin' and y_type == 'bin':
        if np.isin(x[1:], [0,1]).all() and np.isin(y[1:], [0,1]).all():
            corr_val = matthews_corrcoef(x,y)
        else:
            corr_val, p_val = spearmanr(x,y)
    return corr_val

def rank_biserial(x_cont,y_bin):
    """
    Function that calculates the rank biserial correlation coefficient for 2 arrays
    Input x_cont: Continuous numpy array 1
    Input y_bin: Binary numpy array 2
    Output rb_coef: Rank biserial coefficient
    """
    x_cont_rank_y1 = []
    x_cont_rank_y0 = []
    x_cont_rank = rankdata(x_cont)
    for i in range(len(y_bin)):
        if y_bin[i] == 1:
            x_cont_rank_y1.append(x_cont_rank[i])
        else:
            x_cont_rank_y0.append(x_cont_rank[i])
    M1 = np.mean(x_cont_rank_y1)
    M0 = np.mean(x_cont_rank_y0)
    n1 = len(x_cont_rank_y1)
    n0 = len(x_cont_rank_y0)
    rb_coef = 2*(M1 - M0)/(n1+n0)
    return rb_coef

def load_dataset(data_str,train_fraction):
    """
    Function to load all datasets according to data_str and train_fraction
    Input data_str: Name of the dataset to load
    Input train_fraction: Percentage of dataset instances to use as training dataset
    Output train_data: Training dataset
    Output train_data_target: Training dataset labels
    Output test_data: Test dataset
    Output test_data_target: Test dataset labels
    """
    if data_str == 'synth7':
        data = np.genfromtxt(dataset_dir+'/Synthetic7/synthetic7.csv',delimiter=',')
        train_data, train_data_target, test_data, test_data_target = select_train_test_synth(data,train_fraction)
    return train_data, train_data_target, test_data, test_data_target

def select_train_test_synth(data,train_fraction):
    """
    Function that splits data into train and test with their corresponding targets for the synthetic datasets
    Input data: The dataset used for the splits
    Output train_data: Training dataset
    Output train_data_target: The target of the training dataset
    Output test_data: Test dataset
    Output test_data_target: The target of the test dataset
    """
    len_data = len(data)
    range_idx = np.arange(len_data)
    np.random.shuffle(range_idx)
    train_len = int(np.round_(len_data*train_fraction))
    train_range_idx = range_idx[:train_len]
    test_range_idx = range_idx[train_len:]
    train_data = data[train_range_idx,:-1]
    train_data_target = data[train_range_idx,-1]
    test_data = data[test_range_idx,:-1]
    test_data_target = data[test_range_idx,-1]
    return train_data, train_data_target, test_data, test_data_target

def normalization_train(data,data_str):
    """
    Normalization applied to the train dataset on each feature
    Input data: Dataset to be normalized for each feature or column
    Input data_str: String of the dataset's name
    Output normalized_data: Normalized training dataset
    Output train_limits = Normalization parameters for the dataset
    """
    if data_str in ['synth7']:
        max_axis = np.max(data)
        min_axis = np.min(data)
    train_limits = np.vstack((min_axis,max_axis))
    normalized_data = (data - train_limits[0]) / (train_limits[1] - train_limits[0])
    if normalized_data.dtype == 'object':
        normalized_data = normalized_data.astype(float)
    return normalized_data, train_limits

def normalization_test(data,train_limits):
    """
    Normalization applied to the test dataset on each feature
    Input data: Dataset to be normalized for each feature or column
    Input train_limits: the maximum and minimum values per feature from the training dataset
    Output normalized_data: Normalized test dataset
    """ 
    normal_data = data
    normalized_data = (normal_data - train_limits[0]) / (train_limits[1] - train_limits[0])
    normalized_data[normalized_data < 0] = 0
    normalized_data[normalized_data > 1] = 1
    return normalized_data