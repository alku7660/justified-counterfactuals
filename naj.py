"""
Nearest Approximate Justifier Algorithm (NAJ)
"""

"""
Imports
"""

from support import *

def verify_justification(t,x,label,data,data_label,x_label,model,types,n_feat):
    """
    Function that outputs whether the instance x is justified by t instance, with model model
    Input t: Instance 1 (Destination) 
    Input x: Instance 2 (Origin)
    Input label: Label of both instances (same for both)
    Input data: Training dataset 
    Input data_label: Label of the training dataset
    Input x_label: Label of instance of interest x
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input n_feat: Number of instance to generate per feature in the t instance
    Output closest_justifying_instance: Instance that justifies x
    Output justified: 1 or 0, if the instance x is justified by t or not respectively
    Output ratio_bin_justified: Float value indicating whether x is justified by any training instance (it is if ratio_bin_justified > 0) 
    """
    closest_justifying_instance = x
    justified = 0
    ratio_bin_justified = 0
    if (t == x).all() or np.where((x == data).all(axis=1))[0].tolist():
        closest_justifying_instance = t
        justified = 1
        ratio_bin_justified = 1
    else:
        ratio_bin_justified = verify_binary_features_justification(t,x,label,x_label,model,types)
        if ratio_bin_justified > 0:
            closest_justifying_instance, cont_justified = verify_continuous_features_justification(t,x,label,data,data_label,model,types,n_feat)
            if cont_justified:
                justified = 1
    return closest_justifying_instance, justified, ratio_bin_justified

def verify_binary_features_justification(t,x,label,x_label,model,types):
    """
    Function that initially verifies binary feature justification
    Input t: Instance 1 (Destination) 
    Input x: Instance 2 (Origin)
    Input label: Label of both instances (same for both)
    Input x_label: Label of instance of interest x
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Output ratio_justifiability: > 0 if the instance x is justified in its binary features by t or not respectively for any binary path
    """
    vector = t - x
    bin_diff_index = np.where((np.array(types) == 'bin') & (vector != 0))[0].tolist()
    total_perm = math.factorial(len(bin_diff_index))
    previous_perm = []
    possible_perm = []
    if len(bin_diff_index) > 11: # In case there are many permutations, a recursive process is executed to prune and reduce the amount of permutations verified
        possible_perm = recursive_perm_conditional(previous_perm, possible_perm, bin_diff_index, model, label, vector, x)
    else:
        possible_perm = list_perm_conditional(possible_perm, bin_diff_index, model, x_label, vector, x)
    ratio_justifiability = len(possible_perm)/total_perm
    return ratio_justifiability

def recursive_perm_conditional(previous_perm, possible_perm, bin_index, model, label, vector, x):
    """
    Recursively obtain only permutations for which the binary justification is successful
    Input previous_perm: Previous permutation evaluated which was successful
    Input possible_perm: List of all possible index permutations which have been successful
    Input bin_index: Initial binary indexes to change
    Input model: Prediction model used
    Input label: Label of the instance of interest
    Input vector: vector of movement between x and target instance
    Input x: Instance of interest
    Output possible_perm: The list of possible permutations of binary indexes that lead to a binary justified instance of interest
    """
    if len(possible_perm) > 0:
        return possible_perm
    bin_index_new = [i for i in bin_index if i not in previous_perm]
    if len(bin_index_new) == 0:
        possible_perm.append(previous_perm)
        return possible_perm
    if len(previous_perm) > 0:
        for i in range(len(bin_index_new)):
            if len(previous_perm) == 1:
                add = [previous_perm[0],bin_index_new[i]]
            else:
                add = previous_perm.copy()
                add.extend([bin_index_new[i]]) 
            bin_index_new[i] = add
    for i in bin_index_new:
        fail = permutation_verify(x,vector,i,label,model)
        if fail == 0:
            if not isinstance(i,list):
                i = [i]
            possible_perm = recursive_perm_conditional(i, possible_perm, bin_index, model, label, vector, x)
    return possible_perm

def list_perm_conditional(possible_perm, bin_index, model, x_label, vector, x):
    """
    Obtains only permutations for which the binary justification is successful
    Input possible_perm: List of all possible index permutations which have been successful
    Input bin_index: Initial binary indexes to change
    Input model: Prediction model used
    Input label: Label of the instance of interest
    Input x_label: Label of instance of interest x
    Input vector: vector of movement between x and target instance
    Input x: Instance of interest
    Output possible_perm: The list of possible permutations of binary indexes that lead to a binary justified instance of interest
    """
    perm_list = list(permutations(bin_index,len(bin_index)))
    for i in perm_list:
        v = np.copy(x)
        count = 0
        for j in i:
            v[j] += vector[j]
            if verify_same_label(x_label,model,v):
                count += 1
            else:
                break
        if count == len(bin_index):
            possible_perm.append(i)
    return possible_perm

def verify_continuous_features_justification(t,x,label,data,data_label,model,types,n_feat):
    """
    Function that initially verifies continuous feature justification
    Input t: Instance 1 (Destination) 
    Input x: Instance 2 (Origin)
    Input label: Label of both instances (same for both)
    Input data: Training dataset 
    Input data_label: Label of the training dataset
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input n_feat: Number of instance to generate per feature in the t instance
    Output closest_justifying_instance: Instance that justifies x
    Output cont_justified: 1 or 0, if the instance x is justified in its continuous features by t or not respectively
    """
    dist_matrix, all_instances, label_all_instances, type_all_instances, epsilon_scan = continuous_feat_params(t,x,label,data,data_label,model,types,n_feat)
    checked = []
    prev_node = -1
    next_node = 0
    cont_justified = 0
    closest_justifying_instance = x
    instance_chain, cont_justified = chain(x,checked,prev_node,next_node,dist_matrix,label_all_instances,type_all_instances,epsilon_scan,label,cont_justified)
    justifying_tuples = [i for i in instance_chain if 'justified' in i]
    if len(justifying_tuples) > 0:
        justifying_tuples.sort(key=lambda x: x[3])
        closest_justifying_index = justifying_tuples[0][1]
        closest_justifying_instance = all_instances[closest_justifying_index,:]
        cont_justified = 1
    return closest_justifying_instance, cont_justified

def continuous_feat_params(t,x,label,data,data_label,model,types,n_feat):
    """
    Function that outputs parameters needed for continuous feature justification verifying
    Input t: Instance 1 (Destination) 
    Input x: Instance 2 (Origin)
    Input label: Label of both instances (same for both)
    Input data: Training dataset 
    Input data_label: Label from the training data
    Input model: Trained model to verify connectedness between t and x, verifying justification
    Input types: The feature data-type related to each feature in instances
    Input n_feat: Number of instance to generate per feature in the t instance
    Output dist_matrix: Distance matrix among all instances
    Output all_instances: All the instances (np.vstack((x_cont,t,gen_instances)))
    Output label_all_instances: Label of all instances in the matrix
    Output type_all_instances: vector indicating whether the instance is x (x), t (t), from generated instances (g), or from data (d)
    Output epsilon_scan: Distance to check around each instance in the matrix.
    """
    lower = [0]*len(t)
    upper = [1]*len(t)
    x_cont = np.copy(x)
    bin_index = np.where(np.array(types) == 'bin')[0].tolist()
    for i in bin_index:
        lower[i] = t[i]
        upper[i] = t[i]
        x_cont[i] = t[i]
    gen_instances = np.random.uniform(lower,upper,size=(n_feat*len(t),len(t)))
    label_gen_instances = model.predict(gen_instances)
    data_bin_equal, data_bin_equal_label = find_data_equal_bin(data,data_label,bin_index,t)
    if data_bin_equal.shape[0] == 0:
        all_instances = np.vstack((x_cont,t,gen_instances))
        label_all_instances = np.hstack((label,label,label_gen_instances))
        type_all_instances = ['x']+['t']+['g']*len(gen_instances)    
    else:
        all_instances = np.vstack((x_cont,data_bin_equal,t,gen_instances))
        label_all_instances = np.hstack((label,data_bin_equal_label,label,label_gen_instances))
        type_all_instances = ['x']+['d']*len(data_bin_equal)+['t']+['g']*len(gen_instances)
    dist_matrix = distance_matrix(all_instances,all_instances)
    n = len(t)
    epsilon_scan = np.sqrt(types.count('cont'))/5 # This value may be changed for some other value of interest (this was chosen for the results of the r radius study)
    return dist_matrix, all_instances, label_all_instances, type_all_instances, epsilon_scan

def find_data_equal_bin(data,data_label,bin_index,t):
    """
    Function that finds data points in dataset which have same binary feature values as t
    Input data: Training dataset
    Input data: Training dataset label
    Input bin_index: Indexes of binary features
    Input t: Training instance which justifies x
    Output data_equal_bin: Dataset containing only instances that have equal value in binary features between the instance of interest and the training instance
    Output data_equal_bin_label: Label of the instances in data_equal_bin
    """
    data_equal_bin = []
    data_equal_bin_label = []
    for i in range(len(data)):
        counter = 0
        for j in bin_index:
            if data[i,j] == t[j]:
                counter += 1
        if counter == len(bin_index) and not (t == data[i]).all():
            data_equal_bin.append(data[i])
            data_equal_bin_label.append(data_label[i])
    return np.array(data_equal_bin), np.array(data_equal_bin_label)

def chain(x,index_list_checked, index_prev, index_next, dist_matrix, label_all_instances, type_all_instances, epsilon_scan, label, cont_justified):
    """
    Function that creates a chain of paths and finds whether there is a continuous path between the instances. Uses Depth-First search
    Input x: closest CF to the instance of interest and to be verified for justification with the chain
    Input index_list_checked: Set of tuples of interconnected instances
    Input index_prev: instance previously checked
    Input index_next: instance to check next
    Input dist_matrix: Distance matrix among instances
    Input label_all_instances:  The label of each instances in the dist_matrix
    Input type_all_instances: The type of all instances in the dist_matrix
    Input epsilon_scan: The radius of search around each instance
    Input label: Label of the instance and the target instance to justify it with
    Input cont_justified: Variable that becomes 1 when justification is verified
    Output index_list_checked: Set of tuples of interconnected instances
    Output cont_justified: Variable that becomes 1 when justification is verified  
    """
    index_list_checked.append((index_prev,index_next))
    index_prev = index_next
    index_close = np.where((dist_matrix[index_next,:] <= epsilon_scan) & (dist_matrix[index_next,:] > 0))[0]
    for j in index_close:
        if len([i for i in index_list_checked if j in i]) > 0:
            continue
        elif type_all_instances[j] == 'g':
            if label_all_instances[j] != label:
                index_list_checked.append((index_prev,j,'wrong label'))
            else:
                index_next = j
                index_list_checked, cont_justified = chain(x,index_list_checked,index_prev,index_next,dist_matrix,label_all_instances,type_all_instances,epsilon_scan,label,cont_justified)    
                if cont_justified == 1:
                    break
        elif type_all_instances[j] == 'd' or type_all_instances[j] == 't':
            if label_all_instances[j] != label:
                index_list_checked.append((index_prev,j,'wrong label'))
            else:
                index_list_checked.append((index_prev,j,'justified',dist_matrix[0,j]))
                cont_justified = 1
                return index_list_checked, cont_justified
    return index_list_checked, cont_justified

