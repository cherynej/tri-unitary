import sys
import os
import numpy as np
import scipy.sparse as sparse
import json
import warnings
warnings.filterwarnings("ignore")
import time

from SpinLibraryCherynePython3 import *

## What's not in SpinLibraryCherynePython3

def paulixyz():
    I= sparse.csr_matrix([[1., 0.],[0., 1.]]) 
    X= sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    Y= sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    Z= sparse.csr_matrix([[1., 0],[0, -1.]])
    return I,X,Y,Z

I,X,Y,Z = paulixyz()

def mkron(A,B,*argv):
    K=sparse.kron(A,B,'csr')
    for arg in argv: 
        K=sparse.kron(K,arg,'csr')
    return K

### triunitary gate

def UCP_MBL(phi,epsilon,hz,hzz):
    I,X,Y,Z=paulixyz();
    T=translation_op(3);
    VJ_13=T@mkron(I,expm(-1j*(pi/4)*(mkron(X,X)+mkron(Y,Y)+mkron(Z,Z))))@H(T)
    CP12=np.diag([1,1,1,1,1,1,np.exp(1j*phi[0]),np.exp(1j*phi[0])])
    CP23=np.diag([1,1,1,np.exp(1j*phi[1]),1,1,1,np.exp(1j*phi[1])])
    CP31=np.diag([1,1,1,1,1,np.exp(1j*phi[2]),1,np.exp(1j*phi[2])])
    U=VJ_13@CP12@CP23@CP31@expm(-1j*epsilon*(mkron(X,I,I)+mkron(I,X,I)+mkron(I,I,X)))@expm(-1j*(hz[0]*mkron(Z,I,I)+hz[1]*mkron(I,Z,I)+hz[2]*mkron(I,I,Z)))@expm(-1j*(hzz[0]*mkron(Z,Z,I)+hzz[1]*mkron(I,Z,Z)+hzz[2]*mkron(Z,I,Z)))
    return U

### tensor dot evolution

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def tensor_legs_onesite(i,L):
    legs=np.linspace(0,L-1,L,dtype=int).tolist() 
    legs.remove(i)
    legs.insert(0,i)
    return legs

def tensor_op_singlesite(op,vec,i,L):
    dim=2**L
    legs=tensor_legs_onesite(i,L)
    vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
    vec=vec.transpose(legs)
    vec=np.reshape(vec,[2**1,2**(L-1)])
    vec=np.tensordot(op,vec,axes=([1,0]))
    vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
    vec=vec.transpose(inv(legs))
    vec=np.reshape(vec,[dim])
    return vec.transpose()

def tensor_legs_foursite(i,j,k,l,L):
    legs=np.linspace(0,L-1,L,dtype=int).tolist() 
    legs.remove(i)
    legs.remove(j)
    legs.remove(k)
    legs.remove(l)
    legs.insert(0,i)
    legs.insert(1,j)
    legs.insert(2,k)
    legs.insert(3,l)
    return legs

def tensor_op4site_tot_evenlayer(op,vec,L):
    dim=2**L
    vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
    for i in range(0,L-1,4):
        legs=tensor_legs_foursite(i,i+1,i+2,i+3,L)
        vec=vec.transpose(legs)
        vec=np.reshape(vec,[2**4,2**(L-4)])
        vec=np.tensordot(op,vec,axes=([1,0]))
        vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
        vec=vec.transpose(inv(legs))
    vec=np.reshape(vec,[dim])
    return vec.transpose()

def tensor_op4site_tot_oddlayer(op,vec,L):
    dim=2**L
    vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
    for i in range(0,L-1,4):
        legs=tensor_legs_foursite(np.mod(i+2,L),np.mod(i+3,L),np.mod(i+4,L),np.mod(i+5,L),L)
        vec=vec.transpose(legs)
        vec=np.reshape(vec,[2**4,2**(L-4)])
        vec=np.tensordot(op,vec,axes=([1,0]))
        vec=np.reshape(vec,list(2*np.ones(L,dtype=int)))
        vec=vec.transpose(inv(legs))
    vec=np.reshape(vec,[dim])
    return vec.transpose()


    ''' 
        def __init__(self, psi1, psi2, u1234, uto_correlatorZ_nonmover_singleHaar, num, t, n, num_states):
    '''

class Checkpoint:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        version_file = 'version'
        self.version_path = os.path.join(checkpoint_path, version_file)

        if not os.path.exists(self.version_path):
            self.save_version(0)

    def get_version(self):
        with open(self.version_path, 'r') as f:
            line = f.readline()
            return int(line[0])

    def save_version(self, version):
        with open(self.version_path, 'w') as f:
            f.write(str(version))

    def save(self, state):
        new_version = 1 - self.get_version()
        for key in State.fields:
            filepath = os.path.join(checkpoint_path, '_{}_{}.npy'.format(new_version, key))
            np.save(filepath, state[key])
        self.save_version(new_version)

    def load(self):
        current_version = self.get_version()
        try:
            fields_dict = {}
            for f in State.fields:
                print('Loading: ', f)
                fields_dict[f] = np.load(os.path.join(self.checkpoint_path, '_{}_{}.npy'.format(current_version, f)))
        except Exception as e:
            print(e)
            return None
        return State(fields_dict)

def time_evolution_step(psi1, psi2, u1234, L):
    psi2=tensor_op4site_tot_evenlayer(u1234, psi2, L)
    psi2=tensor_op4site_tot_oddlayer(u1234,  psi2, L)

    psi1=tensor_op4site_tot_evenlayer(u1234, psi1, L)
    psi1=tensor_op4site_tot_oddlayer(u1234, psi1, L)
    return psi1, psi2
 
def parse_params_json(params_json):
    params = json.loads(params_json)

    L = params["L"]
    eps = params["eps"]
    W = params["W"]
    W = W*np.pi
    tmax = params["tmax"] #max time
    num_states = params["num_states"] #number of states for canonical typ

    chunk_id = params['chunk_id']
    chunk_id = str(chunk_id).zfill(4)
    chunk_size = params['chunk_size']
    directory = params['directory']
    return L, eps, W, tmax, num_states, chunk_id, chunk_size, directory

def compute_u1234(eps, W):
    phi = np.random.uniform(0,2*np.pi,3)
    hz = np.random.uniform(0,W,3)
    hzz = np.random.uniform(0,W,3)
    u123 = UCP_MBL(phi,eps,hz,hzz)
    u1234 = mkron(u123,I).todense()
    return u1234

def init_psi_Haar(L, num, auto_correlatorZ_nonmover_singleHaar):
    psi=random_state(L)
    psi1=psi
    psi2=psi.copy() # Take deep copy instead of shallow one
    auto_correlatorZ_nonmover_singleHaar[num,0]=np.conj(psi).T@psi
    return psi1, psi2, auto_correlatorZ_nonmover_singleHaar


class State:
    fields = ['nmin', 'num_statesmin', 'tmin', 'auto_correlatorZ_nonmover', 'auto_correlatorZ_nonmover_singleHaar', 'psi1', 'psi2', 'u1234']
    scalars = ['nmin', 'num_statesmin', 'tmin']

    def __init__(self, fields_dict=None):
        '''
            fields_dict: {'psi1': psi1, 'psi2': psi2, ... }
        '''
        if fields_dict:
            for scalar in State.scalars:
                if type(fields_dict[scalar]) is not int:
                    fields_dict[scalar] = int(fields_dict[scalar])
        self.fields_dict = fields_dict 
        

    def __getitem__(self, key):
        return self.fields_dict[key]

    def __setitem__(self, key, val):
        self.fields_dict[key] = val

    def reset(self, L, eps, chunk_size, tmax, num_states):
        dim=2**L
        auto_correlatorZ_nonmover = np.zeros((chunk_size, tmax))
        auto_correlatorZ_nonmover_singleHaar = np.zeros((num_states,tmax),dtype=complex)

        num = 0
        # TODO skipping Haar cause initialized above with 0 
        psi1, psi2, _ = init_psi_Haar(L, num, auto_correlatorZ_nonmover_singleHaar) 

        self.fields_dict = {
                'nmin': 0,
                'num_statesmin': 0,
                'tmin': 0,
                'auto_correlatorZ_nonmover': auto_correlatorZ_nonmover,
                'auto_correlatorZ_nonmover_singleHaar': auto_correlatorZ_nonmover_singleHaar,
                'psi1': psi1,
                'psi2': psi2,
                'u1234': compute_u1234(eps, W)
                }
        return self

############################################################

# Takes as input 5 parameters: L, eps, W, tmax, num_states

params_json = sys.argv[1]
L, eps, W, tmax, num_states, chunk_id, chunk_size, directory = parse_params_json(params_json)
run_identifier = 'L%d_eps%.4f_chunk_id_%s'%(L, eps, chunk_id)
checkpoint_path = os.path.join(directory, run_identifier)

# Initialize state 
checkpoint = Checkpoint(checkpoint_path)
state = checkpoint.load()
if not state: # If failed to read state for the file - reset state
    print("Failed to load checkpoint, reseting state")
    state = State().reset(L, eps, chunk_size, tmax, num_states)

nmin = state['nmin']
num_statesmin = state['num_statesmin']
tmin = state['tmin']
auto_correlatorZ_nonmover = state['auto_correlatorZ_nonmover']
auto_correlatorZ_nonmover_singleHaar = state['auto_correlatorZ_nonmover_singleHaar']
psi1 = state['psi1']
psi2 = state['psi2']
u1234 = state['u1234']

# Here we have init default state OR we have state from checkpoint
print(nmin, num_statesmin, tmin)

for n in range(nmin, chunk_size):
    ## Canonical tipicality, op evolution is local via tensordot
    for num in range(num_statesmin,num_states):
        if num == 0:
            u1234 = compute_u1234(eps, W) 
            auto_correlatorZ_nonmover_singleHaar = np.zeros((num_states,tmax),dtype=complex)

        for t in range(tmin, tmax):
            print(n, num, t)
            # Reset state if time at the beginning
            if t == 0:
                psi1, psi2, auto_correlatorZ_nonmover_singleHaar = init_psi_Haar(L, num, auto_correlatorZ_nonmover_singleHaar)
                psi2=tensor_op_singlesite(Z.todense(),psi2,3,L)

            psi1, psi2 = time_evolution_step(psi1, psi2, u1234, L) 
            auto_correlatorZ_nonmover_singleHaar[num,t]=np.dot(np.conj(tensor_op_singlesite(Z.todense(),psi1,3,L)),psi2)

            # SAVE STATE
            if np.mod(t+1,100)==0:
                state = State({
                    'psi1': psi1, 
                    'psi2': psi2, 
                    'u1234': u1234, 
                    'auto_correlatorZ_nonmover': auto_correlatorZ_nonmover,
                    'auto_correlatorZ_nonmover_singleHaar': auto_correlatorZ_nonmover_singleHaar,
                    'nmin': np.array([n]),
                    'num_statesmin': np.array([num]),
                    'tmin': np.array([(t+1)%tmax])
                    })
                checkpoint.save(state)
        tmin = 0
    num_statesmin = 0
    # SAVE FINAL RESULT
    auto_correlatorZ_nonmover[n,:]=np.real(np.mean(auto_correlatorZ_nonmover_singleHaar,axis=0))
    state['auto_correlatorZ_nonmover'] = auto_correlatorZ_nonmover
    state['nmin']: np.array([n+1])
    state['num_statesmin']: np.array([0])
    state['tmin']: np.array([0])
    checkpoint.save(state)
