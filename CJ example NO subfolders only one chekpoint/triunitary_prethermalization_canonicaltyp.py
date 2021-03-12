import sys
import os
import numpy as np
import scipy.sparse as sparse
import json
import warnings
warnings.filterwarnings("ignore")

from SpinLibraryCherynePython3 import *

## What's not in SpinLibraryCherynePython3

def paulixyz():
    I= sparse.csr_matrix([[1., 0.],[0., 1.]]) 
    X= sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    Y= sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    Z= sparse.csr_matrix([[1., 0],[0, -1.]])
    return I,X,Y,Z


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

############################################################

# Takes as input 5 parameters: L, lam, W, trun, niter
# niter will be read out from checkpoint files

params_json = sys.argv[1]
params = json.loads(params_json)

L = params["L"]
lam = params["lam"]
chunk_id = params['chunk_id']
chunk_id = str(chunk_id).zfill(4)
chunk_size = params['chunk_size']
directory = params['directory']


file_identifier = 'L%d_lam%.4f_chunk_id_%s'%(L, lam, chunk_id)
checkpoint_file = 'checkpoint_%s.npy'%(file_identifier)
checkpoint_path = os.path.join(directory, checkpoint_file)
test_file = 'test_%s.npy'%(file_identifier)
test_path = os.path.join(directory, test_file)


##Setting up
dim=2**L
I,X,Y,Z = paulixyz()
tvec=np.arange(0,tmax,1)
np.save('tvec_L%d.npy' %L, tvec)

        
## Setting up emtpy arrays
auto_correlatorZ_nonmover=np.zeros((chunk_size,len(tvec)))

nmin = 0
if os.path.exists(checkpoint_path):
    try:
        nmin = np.load(checkpoint_path)[0]
        auto_correlatorZ_nonmover[0:nmin] = np.load(test_path)[0:nmin,:]
        print(nmin)
    except:
        nmin=0
    

for n in range(nmin,niter):
    #print(runs)
    phi=np.random.uniform(0,2*np.pi,3)
    hz=np.random.uniform(0,W,3)
    hzz=np.random.uniform(0,W,3)
    u123=UCP_MBL(phi,eps,hz,hzz)
    u1234=mkron(u123,I)
    
    ## Canonical tipicality, op evolution is local via tensordot
    auto_correlatorZ_nonnmover_singleHaar=np.zeros((num_states,len(tvec)),dtype=complex)

    for num in range(num_states):
        psi=random_state(L)
        psi1=psi
        psi2=psi
        auto_correlatorZ_nonnmover_singleHaar[num,0]=np.conj(psi).T@psi
        psi2=tensor_op_singlesite(Z.todense(),psi2,3,L)
        for t in range(1,len(tvec)):
            psi2=tensor_op4site_tot_evenlayer(u1234.todense(),psi2,L)
            psi2=tensor_op4site_tot_oddlayer(u1234.todense(),psi2,L)
            
            psi1=tensor_op4site_tot_evenlayer(u1234.todense(),psi1,L)
            psi1=tensor_op4site_tot_oddlayer(u1234.todense(),psi1,L)
            
            auto_correlatorZ_nonnmover_singleHaar[num,t]=np.dot(np.conj(tensor_op_singlesite(Z.todense(),psi1,3,L)),psi2)
            
    auto_correlatorZ_nonmover[n,:]=np.real(np.mean(auto_correlatorZ_nonnmover_singleHaar,axis=0))
    np.save(test_path,auto_correlatorZ_nonmover[0:n+1,:])
    np.save(checkpoint_path, np.array([n+1]))
