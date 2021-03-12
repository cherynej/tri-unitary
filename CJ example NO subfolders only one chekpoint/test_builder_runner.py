import sys
import os
import numpy as np

#params = [
#        '10 0.1 2.0 5 10 12',
#        '11 0.1 2.0 5 10 12',
#        '12 0.1 2.0 5 10 12'
#        ]

Lvec = [16,22]
epsvec = np.concatenate([np.arange(0,0.1,0.005),np.arange(0.1,1,0.05)])
Wintvec = [2]
num_statesvec = [5]
tmaxvec = [10**7]
nitervec = [1,1]

params=[]

Wint=Wintvec[0]
num_states=num_statesvec[0]
tmax=tmaxvec[0]

for ind_L,L in enumerate(Lvec):
    for ind_e,eps in enumerate(epsvec):
        for ind_n,niter in enumerate(nitervec):
            params.append('%d %.6f %d %d %d %d' %(L,eps,Wint,num_states,tmax,nitervec[ind_L]))

num_runs = 100

for p in params:
    for i in range(num_runs):
        os.system('python test_builder.py ' + p)
