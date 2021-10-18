"""
Direct Justified Neighbor (DJN)
"""

"""
Imports
"""

from naj import *

def nt(x,x_label,data_distance):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data_distance: Training dataset organized by distance to x
    Output nt_cf: Minimum observable counterfactual to the instance of interest x
    Output nt_cf_dist_x: Distance between nt_cf and instance of interest x
    """
    for i in data_distance:
        if i[2] != x_label:
            nt_cf = i[0]
            nt_cf_dist_x = i[1]
            break
    return nt_cf, nt_cf_dist_x

def direct_justified_nn(x,x_label,data_distance,model,types,step):
    """
    Direct justified NN counterfactual generation method:
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data_distance: Training dataset organized by distance to x
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Output justified_CF: Closest Counterfactual instance justified by CF_NT
    Output justified_CF_dist_x: Justified CF instance distance to x
    Output closest_knn_cf: Justifier instance to closest_justified_CF
    Output justified: Binary value indicating whether CF is justified or not
    Output bin_just_ratio: Binary value indicating whether closest_justified_CF is binary justified
    """
    justified = 0
    closest_knn_cf, closest_knn_cf_dist = nt(x,x_label,data_distance)
    if closest_knn_cf_dist > 0:
        justified = 1
    justified_CF, justified_CF_dist_x, bin_just_ratio = justified_search(x,x_label,closest_knn_cf,model,types,step)
    return justified_CF, justified_CF_dist_x, closest_knn_cf, justified, bin_just_ratio

def justified_search(x,x_label,t,model,types,step):
    """
    Search for instances justified by t (train instance), closer to instance x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input t: Training instance
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Output cont_closest_to_x: Instance that is justified by t
    Output cont_closest_to_x_dist: Justified CF instance distance to x
    Output bin_just_ratio: Binary value indicating whether closest_justified_CF is binary justified 
    """
    bin_closest_to_x, bin_just_ratio = binary_justified_search(x,x_label,t,model,types)
    cont_closest_to_x, cont_closest_to_x_dist = continuous_justified_search(x,x_label,bin_closest_to_x,model,step,types)
    return cont_closest_to_x, cont_closest_to_x_dist, bin_just_ratio

def binary_justified_search(x,x_label,t,model,types):
    """
    Search for instances closer to x by modifying only binary features
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input t: Training instance
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Output closest_to_x: Closest instance to x that is justified by t
    Output bin_just_ratio: Binary value indicating whether closest_justified_CF is binary justified 
    """
    vector = x - t
    closest_to_x = t
    closest_to_x_dist = np.sqrt(np.sum((x-closest_to_x)**2))
    bin_diff_index = np.where((np.array(types) == 'bin') & (vector != 0))[0]
    # if data_str == 'Hepatitis' or data_str == 'Heart Disease' and 1 in bin_diff_index:
    #     bin_diff_index = np.delete(bin_diff_index,np.where(bin_diff_index == 1))
    perm_list = list(permutations(bin_diff_index,len(bin_diff_index)))
    for i in perm_list:
        v = np.copy(t)
        for j in i:
            v[j] += vector[j]
            # Different label from x, since we are in direct justified NN, looking for closest point to x, that is NOT from its class and is justified by t
            if verify_diff_label(x_label,model,v) and np.sqrt(np.sum((x-v)**2)) < closest_to_x_dist:
                closest_to_x = np.copy(v)
                closest_to_x_dist = np.copy(np.sqrt(np.sum((x-v)**2)))
            else:
                break
    bin_just_ratio = verify_binary_features_justification(t,closest_to_x,int(1 - x_label),x_label,model,types)
    return closest_to_x, bin_just_ratio

def continuous_justified_search(x,x_label,bin_closest_to_x,model,step,types):
    """
    Search for instances closer to x by modifying only binary features
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input bin_closest_to_x: closest instance to x so far still belonging to counterfactual class and connected binarily to t
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Output closest_to_x: Instance that is justified by t
    Output closest_to_x_dist: Justified CF instance distance to x
    """
    v = np.copy(bin_closest_to_x)
    vector = x - bin_closest_to_x
    closest_to_x = v
    closest_to_x_dist = np.sqrt(np.sum((x-closest_to_x)**2))
    cont_diff_index = np.where((np.array(types) == 'cont') & (vector != 0))[0].tolist()
    iterations = int(1/step)
    for i in range(iterations):
        v[cont_diff_index] += vector[cont_diff_index]*step
        # Different label from x, since we are in direct justified NN, looking for closest point to x, that is NOT from its class and is justified by t
        if verify_diff_label(x_label,model,v) and np.sqrt(np.sum((x-v)**2)) < closest_to_x_dist:
            closest_to_x = np.copy(v)
            closest_to_x_dist = np.copy(np.sqrt(np.sum((x-v)**2)))
        else:
            v[cont_diff_index] -= vector[cont_diff_index]*step
            break
    close_vector = x - closest_to_x
    close_cont_diff_index = np.where((np.array(types) == 'cont') & (close_vector != 0))[0].tolist()
    # if data_str == 'Diabetes' or data_str == 'Hepatitis' or data_str == 'Heart Disease' or data_str == 'Cervical Cancer' and 0 in close_cont_diff_index:
    #     close_cont_diff_index = np.delete(close_cont_diff_index,np.where(np.array(close_cont_diff_index) == 0)).tolist()
    # if data_str == 'Cervical Cancer' and 1 in close_cont_diff_index:
    #     close_cont_diff_index = np.delete(close_cont_diff_index,np.where(np.array(close_cont_diff_index) == 1)).tolist()
    closest_to_x, closest_to_x_dist = single_continuous_justified_search(close_cont_diff_index,x,x_label,closest_to_x,model,close_vector,step)
    return closest_to_x, closest_to_x_dist

def single_continuous_justified_search(cont_diff_index,x,x_label,v,model,vector,step):
    """
    Search for instances closer to x by modifying single continuous features
    Input cont_diff_index: Continuous feature indexes
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input v: current instance
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input vector: Current vector from v to x
    Input step: Percentual step to apply to numerical variables (0.01 recommended)
    Output closest_to_x: Instance that is justified by t
    Output closest_to_x_dist: Justified CF instance distance to x
    """
    closest_to_x = v
    closest_to_x_dist = np.sqrt(np.sum((x-closest_to_x)**2))
    for i in cont_diff_index:
        can = True
        while can:
            v[i] += vector[i]*step
            # Different label from x, since we are in direct justified NN, looking for closest point to x, that is NOT from its class and is justified by t
            if verify_diff_label(x_label,model,v) and np.sqrt(np.sum((x-v)**2)) < closest_to_x_dist:
                closest_to_x = np.copy(v)
                closest_to_x_dist = np.copy(np.sqrt(np.sum((x-v)**2)))
            else:
                v[i] -= vector[i]*step
                can = False
            if v[i] <= 0 or v[i] >= 1: 
                can = False
    return closest_to_x, closest_to_x_dist