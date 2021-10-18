"""
Correlation Indirect Justified Neighbor (CIJN)
"""

"""
Imports
"""

from naj import *
from djn import *

def corr_indirect_justified_nn(x,x_label,data,data_str,k,data_label,data_distance,model,types,step,n_feat):
    """
    Indirect justified NN counterfactual generation method:
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Training dataset 
    Input data_str: String corresponding to the identifier of the dataset
    Input k: Number of training dataset neighbors to consider for the correlation matrix calculation
    Input data_label: Label of the training dataset instances
    Input data_distance: Training dataset organized by distance to x
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Input n_feat: Number of instance to generate per feature in the t instance
    Output closest_justified_CF: Closest Counterfactual instance justified
    Output justified_CF_dist_x: Distance to closest_justified_CF from x
    Output CF_justifying_instance: Justifier instance to closest_justified_CF
    Output CF_justified: Binary value indicating whether CF is justified or not
    Output use_direct: Binary value indicating whether DJN was used
    Output bin_just_ratio: Binary value indicating whether closest_justified_CF is binary justified
    """
    closest_justified_cf = 0
    CF_justifying_instance = 0
    closest_justified_cf_dist_x = 0
    CF_justified = 0
    use_direct = 0
    closest_knn_cf, closest_knn_cf_dist = nt(x, x_label,data_distance)
    v_list = corr_feature_change(x,x_label,data_distance,k,closest_knn_cf,model,types,step)
    if len(v_list) == 0:
        justified_CF, justified_CF_dist_x, CF_justifying_instance, CF_justified, bin_just_ratio = direct_justified_nn(x,x_label,data_distance,model,types,step)
        closest_justified_cf = justified_CF
        closest_justified_cf_dist_x = justified_CF_dist_x
        use_direct = 1
    else:
        v_array = np.array([i[1] for i in v_list])
        v_label = model.predict(v_array)
        unique_v_label = np.unique(v_label)
        #Verifies if all single feature change CF have the same counterfactual label and are opposite to the instance label
        if len(unique_v_label) == 1 and unique_v_label != x_label:
            x_v_knn = sort_data_distance(x,v_array,v_label)
        else:
            print('CF found by corr feature change are not from the opposite class or are different between them')
        for j in x_v_knn:
            nn_j, label_nn_j = nn_to_cf(j[0], data, data_label, x_label, data_str)
            CF_justifier, CF_justification, bin_just_ratio = verify_justification(nn_j,j[0],unique_v_label,data,data_label,x_label,model,types,n_feat)
            if CF_justification:
                closest_justified_cf = j[0]
                closest_justified_cf_dist_x = j[1]
                CF_justifying_instance = CF_justifier
                CF_justified = CF_justification
                break
        if CF_justified == 0:
            justified_CF, justified_CF_dist_x, CF_justifying_instance, CF_justified, bin_just_ratio = direct_justified_nn(x,x_label,data_distance,model,types,step)
            closest_justified_cf = justified_CF
            closest_justified_cf_dist_x = justified_CF_dist_x
            use_direct = 1
            CF_justified = 1
    return closest_justified_cf, closest_justified_cf_dist_x, CF_justifying_instance, CF_justified, use_direct, bin_just_ratio

def corr_feature_change(x,x_label,data_distance,k,t,model,types,step):
    """
    Function that creates correlation aware feature gradual and valid changes to be evaluated and checked with the model
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data_distance: Training dataset organized by distance to x
    Input k: Number of training dataset neighbors to consider for the correlation matrix calculation
    Input t: Training instance
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Output v_list: list of instances belonging to counterfactual class obtained by single-feature gradual and valid changes
    """
    x_NN = [i[0] for i in data_distance]
    x_kNN = x_NN[:k]
    c_matrix = corr_matrix(np.array(x_kNN),types)
    vector = t - x 
    bin_diff_index = np.where((np.array(types) == 'bin') & (vector != 0))[0].tolist()
    cont_diff_index = np.where((np.array(types) == 'cont') & (vector != 0))[0].tolist()
    # if data_str == 'Hepatitis' or data_str == 'Heart Disease' and 1 in bin_diff_index:
    #     bin_diff_index = np.delete(bin_diff_index,np.where(np.array(bin_diff_index) == 1)).tolist()
    # if data_str == 'Diabetes' or data_str == 'Hepatitis' or data_str == 'Heart Disease' or data_str == 'Cervical Cancer' and 0 in cont_diff_index:
    #     cont_diff_index = np.delete(cont_diff_index,np.where(np.array(cont_diff_index) == 0)).tolist()
    # if data_str == 'Cervical Cancer' and 1 in cont_diff_index:
    #     cont_diff_index = np.delete(cont_diff_index,np.where(np.array(cont_diff_index) == 1)).tolist()
    v_list = []
    v = np.copy(x)
    if len(bin_diff_index) > 0:
        for i in bin_diff_index:
            v = np.copy(x)
            v += vector[i]*c_matrix[i,:]
            v = validate_instance(v,types)
            if verify_diff_label(x_label,model,v):
                v_list.append((i,v))
    if len(cont_diff_index) > 0:
        for i in cont_diff_index:
            v = np.copy(x)
            can = True
            while can:
                v += vector[i]*step*c_matrix[i,:]
                v = validate_instance(v,types)
                if verify_diff_label(x_label,model,v):
                    v_list.append((i,v))
                    can = False
                if v[i] <= 0 or v[i] >= 1: 
                    can = False 
    return v_list

def validate_instance(v,types):
    """
    Auxiliary function that transforms an instance v into a valid instance
    Input v: Instance of interest
    Input types: Types of variables in v
    Output v: Validated instance
    """
    for i in range(len(v)):
        if types[i] == 'bin':
            v[i] = np.around(v[i])
            if v[i] < 0:
                v[i] = 0
            elif v[i] > 1:
                v[i] = 1
        else:
            if v[i] < 0:
                v[i] = 0
            elif v[i] > 1:
                v[i] = 1
    return v    