import os
import sys
import pandas as pd
import json
import numpy as np
import csv

CONFIG_PATH='config.csv'

def generate_config():
    Lvec=[20] #system size
    #epsvec = [0.05,0.8] #eps vector, X-term
    epsvec = np.concatenate([np.arange(0,0.1,0.005),np.arange(0.1,1,0.05)])
    Wintvec = [2] #width of disorder in Z-terms, integer to be multiplied by pi in code
    num_statesvec = [5] #number of states to calculate canonical typicality with
    tmaxvec=[1000000]
	#tmaxvec=[10000] #max time
    niter=10 #total number of iterations
    chunk_size=1 #break them into chunks
    
    params=[]

    Wint=Wintvec[0]
    num_states=num_statesvec[0]
    tmax=tmaxvec[0]

    header = ["L", "eps", "W", "tmax", "num_states", "niter", "chunk_size"]
    with open(CONFIG_PATH, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(header)
        for ind_L,L in enumerate(Lvec):
            for ind_e,eps in enumerate(epsvec):
                writer.writerow([L, eps, Wint, tmax, num_states, niter, chunk_size])

directory = sys.argv[1]

if not os.path.exists(directory):
    try:
        os.mkdir(directory)
    except OSError:
        print ("Creation of the directory %s failed" % directory)

generate_config()
config = pd.read_csv(CONFIG_PATH)
for index, row in config.iterrows():
    params = {col:row[col] for col in config.columns}
    params['directory'] = directory
    niter = int(params["niter"])
    chunk_size = int(params["chunk_size"])
    num_jobs = niter // chunk_size 
    for i in range(num_jobs):
        params['chunk_id'] = i
        params_string = json.dumps(params, separators=(',',':'))
        command = 'sbatch cheryne.qusub \'{}\''.format(params_string)
        os.system(command)

    if niter % chunk_size != 0:
        params['chunk_id'] = num_jobs
        params['chunk_size'] = niter % chunk_size
        params_string = json.dumps(params, separators=(',',':'))
        command = 'sbatch cheryne.qsub \'{}\''.format(params_string)
        os.system(command)
