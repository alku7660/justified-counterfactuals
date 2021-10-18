"""
Imports
"""

from naj import *
from djn import *
from ijn import *
from cijn import *

types = ['cont', 'cont']   # Define the nature of the features in the dataset
data_str = 'synth7'        # Name of the dataset to be analyzed (Change lines 14 and 15 in support.py to point to dataset directory)
step = 0.2                 # step size to change continuous features
train_fraction = 0.8       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
k = 50                     # Number of training dataset neighbors to consider for the correlation matrix calculation for the CIJN method

seed_int = 20
np.random.seed(seed_int)
train_data, train_data_target, test_data, test_data_target = load_dataset(data_str,train_fraction)
normal_train_data, train_limits = normalization_train(train_data,data_str)
normal_test_data = normalization_test(test_data,train_limits)
random_idx = np.random.randint(0,len(normal_test_data))
x = normal_test_data[random_idx]
data_distance = sort_data_distance(x,normal_train_data,train_data_target)

# Model may be changed to any classifier
lin_svm_model = svm.SVC(kernel='linear')
lin_svm_model.fit(normal_train_data,train_data_target)
x_label = lin_svm_model.predict(x.reshape(1,-1))

# Counterfactuals and Justifiers Calculation (Refer to naj.py for the NAJ algorithm)
nt_cf, nt_cf_dist_x  = nt(x,x_label,data_distance)
djn_cf, djn_cf_dist_x, instance_djn, just_djn, bin_just_ratio_djn = direct_justified_nn(x,x_label,data_distance,lin_svm_model,types,step) # Refer to djn.py for details
ijn_cf, ijn_cf_dist_x, instance_ijn, just_ijn, used_direct_ijn, bin_just_ratio_ijn = indirect_justified_nn(x,x_label,normal_train_data,data_str,train_data_target,data_distance,lin_svm_model,types,step,n_feat) # Refer to ijn.py for details
cijn_cf, cijn_cf_dist_x, instance_cijn, just_cijn, used_direct_cijn, bin_just_ratio_cijn = corr_indirect_justified_nn(x,x_label,normal_train_data,data_str,k,train_data_target,data_distance,lin_svm_model,types,step,n_feat) # Refer to cijn.py for details

train_limits_range = train_limits[1,:] - train_limits[0,:]
x = x*train_limits_range + train_limits[0,:]
nt_cf = nt_cf*train_limits_range + train_limits[0,:]
djn_cf = djn_cf*train_limits_range + train_limits[0,:]
instance_djn = instance_djn*train_limits_range + train_limits[0,:]
ijn_cf = ijn_cf*train_limits_range + train_limits[0,:]
instance_ijn = instance_ijn*train_limits_range + train_limits[0,:]
cijn_cf = cijn_cf*train_limits_range + train_limits[0,:]
instance_cijn = instance_cijn*train_limits_range + train_limits[0,:]

print(f'----------------------------------------------------------------------')
print(f'----------------------- Found Counterfactuals: -----------------------')
print(f'             Instance of Interest (x):{x}')
print(f'nt_cf    :{nt_cf}, justifier:{nt_cf}')
print(f'djn_cf   :{djn_cf}, justifier:{instance_djn}')
print(f'ijn_cf   :{ijn_cf}, justifier:{instance_ijn}')
print(f'cijn_cf  :{cijn_cf}, justifier:{instance_cijn}')
print(f'----------------------------------------------------------------------')

