import sys
import os
import numpy as np
import scipy.sparse as sparse
import json
import warnings
warnings.filterwarnings("ignore")
import time
import glob
import sys

L=12
Wint=2
#epsvec=[0.02]
epsvec = np.concatenate([np.arange(0.,0.1,0.005),np.arange(0.1,1,0.05)])

for eps in epsvec:
	fnames_auto_correlator = glob.glob('./dir1_L12/L12_eps%.4f*/*auto_correlatorZ_nonmover.npy' %eps)
	print(fnames_auto_correlator)
	print(len(fnames_auto_correlator))
	
	if len(fnames_auto_correlator) == 0:
		print(['eps = %.4f does not exist'% eps])
	else:
		count_fail=0
		count_success=0
		auto_correlator_data_assembly=[]
		for i in range(len(fnames_auto_correlator)):
			path = fnames_auto_correlator[i]
			tokens = path.split('/')
			# Construct version path
			auto_correlator_name = tokens[-1]
			tokens[-1] = 'version'
			version_path = '/'.join(tokens)
			# Read version from file
			version = None
			with open(version_path) as f:
				version = f.read()
			# Ignore file if version prefix does not match
			if auto_correlator_name[0] != version:
				continue
			
			print(fnames_auto_correlator[i])
			auto_correlator=np.load(fnames_auto_correlator[i])
			if auto_correlator[0,0]==0.:
				count_fail+=1
				#print('code %d didnt complete' %i)
			else:
				auto_correlator_data_assembly.append(auto_correlator)
				#print('added')
				count_success+=1

	print(['count_fail=' ,count_fail])
	print(['count_success=', count_success])
	np.save('auto_correlator_saved_mean_L%d_eps%.4f' %(L,eps), np.mean(np.squeeze(auto_correlator_data_assembly),axis=0))
	#sys.getsizeof(auto_correlator_data_assembly)
	#np.save('auto_correlator_saved_L%d_eps%.4f' %(L,eps), auto_correlator_data_assembly)
