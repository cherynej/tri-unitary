#!/usr/bin/env python
# coding: utf-8

# In[1]:

import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np
import itertools
from scipy import linalg
import os
import sys
from itertools import chain, combinations
import scipy
import time
import operator
import functools
import copy
import string
import matplotlib.pylab as plt
from scipy.special import comb
from scipy.linalg import expm, sinm, cosm
from scipy.sparse.linalg import eigs

import numpy as np
from numpy import linalg as LA
from numpy import pi as pi
import matplotlib.pyplot as plt

import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin

from scipy.special import comb
from scipy.linalg import expm, sinm, cosm
from scipy.sparse.linalg import eigs

import sys
import seaborn as sns
import copy
from itertools import combinations
#from sympy import symbols
#from sympy import *
import itertools
from itertools import chain, combinations

import time
import operator
import functools
#import string

######### MISCELLANEOUS MATRIX TRICKS #########

def H(mat):
    ctranspose = mat.conj().T
    return ctranspose

def reordermat(pattern): 
    pattern=np.array(pattern)-1
    N=len(pattern)
    m=np.zeros((2**N,2**N))
    a=np.zeros(N)
    n=0
    while True:
        m[n,np.int(np.sum(2**np.array(pattern)*a))]=1
        k=N-1;
        a[k]=a[k]+1;
        while np.int(a[k])==2:
            a[k]=0;
            k=k-1;
            if k==-1:
                return m
            a[k]=a[k]+1;
        n=n+1

def mkron(A,B,*argv):
    K=sparse.kron(A,B,'csr')
    for arg in argv: 
        K=sparse.kron(K,arg,'csr')
    return K

def multikron(op):
    out = op[0]
    for i in range(1, len(op)):
        out = np.kron(out, op[i])
    return out

def pkron(A,no):
    if no == 0:
        [a,b]=A.shape
        m=np.eye(a,b)
    else:
        m=A
        for n in range(no-1):
            m=sparse.kron(m,A,'csr')
    return m

def paulixyz():
    I= sparse.csr_matrix([[1., 0.],[0., 1.]]) 
    X= sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    Y= sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    Z= sparse.csr_matrix([[1., 0],[0, -1.]])
    return I,X,Y,Z

def printmatrix(matrix):
    s = [[str(np.round(e,3)) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def plot_matrix_structure(H):
    plt.imshow(np.where(np.abs(H)>1e-5,1,0),cmap='Greys')
    plt.xlabel(r'index $i$')
    plt.ylabel(r'index $j$')
    plt.title(r'$H_{ij}$')
    plt.show()

######### GENERATE SPIN HAMILTONIANS #########

def gen_s0sxsysz(L): 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    sy = sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    I = sparse.csr_matrix(np.eye(2**L))
    for i_site in range(L):
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(2)) 
            Y= sparse.csr_matrix(np.eye(2)) 
            Z= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(2),'csr') 
                Y=sparse.kron(Y,np.eye(2),'csr') 
                Z=sparse.kron(Z,np.eye(2),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        s0_list.append(I)

    return s0_list, sx_list,sy_list,sz_list 


def gen_op_total(op_list):
    L = len(op_list)
    tot = op_list[0]
    for i in range(1,L): 
        tot = tot + op_list[i] 
    return tot

def gen_onsite_field(op_list, h_list):
    L= len(op_list)
    H = h_list[0]*op_list[0]
    for i in range(1,L): 
        H = H + h_list[i]*op_list[i] 
    return H

# generates \sum_i O_i O_{i+k} type interactions
def gen_interaction_kdist(op_list, op_list2=[],k=1, J_list=[], bc='obc'):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    H = sparse.csr_matrix(op_list[0].shape)
    if J_list == []:
        J_list =[1]*L
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        H = H+ J_list[i]*op_list[i]*op_list2[np.mod(i+k,L)]
    return H        
    
def gen_nn_int(op_list, J_list=[], bc='obc'):
    return gen_interaction_kdist(op_list,op_list, 1, J_list, bc)

# generates \sum_i O_i O_{i+1} O_{i+2} O_{i+3} type interactions
def gen_interaction_3ops(op_list, op_list2=[],op_list3=[], J_list=[], bc='obc'):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    
    if op_list3 ==[]:
        op_list3=op_list

    H = sparse.csr_matrix(op_list[0].shape)
    
    if J_list == []:
        J_list =[1]*L
        
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        H = H+ J_list[i]*op_list[i]*op_list2[np.mod(i+1,L)]*op_list3[np.mod(i+2,L)]
    return H 

# generates \sum_i O_i O_{i+1} O_{i+2} O_{i+3} type interactions
def gen_interaction_4ops(op_list, op_list2=[],op_list3=[],op_list4=[], J_list=[], bc='obc'):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    
    if op_list3 ==[]:
        op_list3=op_list
        
    if op_list4 ==[]:
        op_list4=op_list
        
    H = sparse.csr_matrix(op_list[0].shape)
    
    if J_list == []:
        J_list =[1]*L
        
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
           H = H+ J_list[i]*op_list[i]*op_list2[np.mod(i+1,L)]*op_list3[np.mod(i+2,L)]*op_list4[np.mod(i+3,L)]
    return H
             


def gen_diagprojector(symvec, symval):
    ind = np.where(symvec==float(symval))
    dim = np.size(ind)
    P = sparse.lil_matrix((dim,len(symvec)))
    for j in range(dim):
        P[j,ind[0][j]] = 1.0
    return P


######### GENERATE PROBABILITY DISTRIBUTIONS FOR COUPLINGS #########

# Power law coupling code
def power_law(x,alpha):
    f= alpha/abs(x)**(1-alpha)
    return f
    
def powerlaw_distribution(alpha,N):
    
    s=list();
    a=1000;
    
    while len(s)<N:
        x0=np.random.uniform(-1.0, 1.0, 1);
        y0=np.random.uniform(0, a, 1);
        if y0 < power_law(x0,alpha):
            s.append(x0)
            
    s=np.asarray(s)  
    s=s[:,0]
    return s

######### COMPUTE QUANTITIES #########

def LevelStatistics(energySpec, ret=True):
    energySpec=np.sort(energySpec)
    delta = energySpec[1:] -energySpec[0:-1]
    r = list(map(lambda x,y: min(x,y)*1.0/max(x,y), delta[1:], delta[0:-1]))
    if ret==True:
        return np.array(delta), np.array(r), np.mean(r)
    return np.mean(r)


def ratio_mean(evals):
    #calculate all the level statistics
    evals=np.sort(evals)
    delta = evals[1:] -evals[0:-1]
    r = list(map(lambda x,y: min(x,y)*1.0/max(x,y), delta[1:], delta[0:-1]))
    
    return np.mean(r)

def reduced_density_matrix(rho_reordered,Nr):
    sx, sy = rho_reordered.shape
    L=np.log2(sx)/np.log2(2)
    L=np.int(np.floor(L+0.5))
    rho_red=np.zeros((2**Nr,2**Nr),dtype=complex)
    for k in range(2**Nr):
        for l in range (2**Nr):
            rr=0
            i1=k*2**(L-Nr)
            i2=l*2**(L-Nr)
            for n in range(2**(L-Nr)):
                rr=rr+rho_reordered[i1+n,i2+n]
            rho_red[k,l]=rr
    return rho_red

def EntanglementEntropy(state, cut_x,dim=2):
    L = int(np.round(np.log(len(state))/np.log(dim)))
    Cij = np.reshape(state, (dim**cut_x, dim**(L-cut_x)))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    S= abs(S)
    S = S[S>(10**-15)]
    return - np.sum((S**2)*np.log2(S**2))

def sff(D,tvec):
    ZZ=np.zeros(len(tvec))
    for ind,t in enumerate(tvec):
        Z= np.sum(D**t)
        ZZ[ind]=np.real(Z*Z.conj())
    return ZZ

    
def RPM(b,N,t):
    vec=2*t*(1+np.cos(2*b)**t)**N
    return vec

def sff_smearing(vec,window):
    smeared_vec=[]
    for i in range(window, len(vec) - window):
        local_mean = 0
        for j in range(-window,window+1):
            local_mean += vec[i+j]
        local_mean = local_mean/(2*window + 1)
        smeared_vec.append(local_mean)
    return smeared_vec


def GOE_SFF(t, N):
    t = np.float64(t)
    return np.piecewise(t, [t < N, t >= N], [2*t - t * np.log(1+ 2*t/N), N])
                        
def GUE_SFF(t, N):
    t = np.float64(t)
    return np.piecewise(t, [t < N, t >= N], [t, N])


######### GENERATE RANDOM STATES AND GATES #########

def random_state(L):
    #Random normalized state
    Psi=np.random.rand(2**L,1)
    Psi=Psi/linalg.norm(Psi)
    return Psi

def random_product_state(L):
    #Product state of each spin pointing randomly on Bloch sphere, 
    #has zero entanglement across all bipartitions
    Psi=np.random.rand(2,1)
    for i in range(L-1):
        Psi=mkron(Psi,np.random.rand(2,1))
    Psi=Psi.toarray()
    Psi=Psi/linalg.norm(Psi);
    return Psi

def gen_randU(n):
    X = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2);
    [Q,R] = np.linalg.qr(X);
    R = np.diag(np.diag(R)/abs(np.diag(R)));
    return np.dot(Q,R)

def rherm(dim):
    A=np.random.normal(0,1,(dim,dim))+1j*np.random.normal(0,1,(dim,dim))
    H=(A+np.conj(A).T)/2
    H=H*np.sqrt(1/np.trace(H@H)) #So that matrix has trace 1 i.e. is "normalized"
    return H


######### Parity Symmetry #########

def get_P(L):
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    P = sx
    for i in range(L-1):
        P =sparse.kron(sx,P)
    return P

def get_Podd(L):
    I = sparse.csr_matrix([[1., 0.],[0., 1.]]) 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    P = mkron(I,sx)
    for i in range(2,L-1,2):
        P =mkron(I,sx,P)
    return P

def get_Peven(L):
    I = sparse.csr_matrix([[1., 0.],[0., 1.]]) 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    P = mkron(sx,I)
    for i in range(2,L-1,2):
        P =mkron(sx,I,P)
    return P


######### SU2 2 and 3 spin blocks #########

def su2_2spinblock():
    phi1=np.random.uniform(0,2*pi,1)
    phi2=np.random.uniform(0,2*pi,1)
    R=np.matrix([[1,0,0,0],[0,1/sqrt(2),1/sqrt(2),0],[0,0,0,1],[0,1/sqrt(2),-1/sqrt(2),0]])
    Ust=block_diag(np.exp(1j*phi1)*np.eye(3),np.exp(1j*phi2))
    U=np.dot(np.dot(R.H,Ust),R)
    return U

def su2_3spinblock():
    vv=np.zeros((8,8));
    vv[0,7]=1
    vv[1,3]=1/sqrt(3);
    vv[1,5]=1/sqrt(3);
    vv[1,6]=1/sqrt(3);

    vv[2,1]=1/sqrt(3);
    vv[2,2]=1/sqrt(3);
    vv[2,4]=1/sqrt(3);

    vv[3,0]=1;

    vv[4,1]=-1/sqrt(2);
    vv[4,4]=1/sqrt(2);

    vv[5,1]=-1/sqrt(6);
    vv[5,2]=sqrt(2/3);
    vv[5,4]=-1/sqrt(6);
    
    vv[6,3]=-1/sqrt(2);
    vv[6,6]=1/sqrt(2);

    vv[7,3]=1/sqrt(6);
    vv[7,5]=-sqrt(2/3);
    vv[7,6]=1/sqrt(6);

    vv=np.matrix(vv.T);
    
    phi=np.random.uniform(0,2*pi,1)
    M=block_diag(np.exp(1j*phi)*np.eye(4),mkron(np.eye(2),gen_randU(2)).toarray());
    Mtilde=np.dot(np.dot(vv,M),vv.H);
    return Mtilde


######### FROM BITSTRINGS TO INTEGERS TO STATES #########
    
    
def bin2int(b):
    L = len(b)
    n = 0
    for i in range(L):
        if b[i] == '1':
            n += 2 **(L - i - 1)
    return n

def de2bi(n,L):
    a=bin(n)[2:].zfill(L)
    a=np.array(list(a), dtype=int)
    return a


def int_to_state(n, L):
   #'''generates the n'th parity eigenstate
    # by taking binary representation of n and
   # translating 0 -> |+> and 1 -> |->'''
    b = np.binary_repr(n,width=L)
    state = [1,0] if b[0]=='1' else [0,1]
    for j in range(1,L):
        if b[j]=='1':
            state = sparse.kron(state,[1,0],'csc')
        else:
            state = sparse.kron(state,[0,1],'csc')
    return state


#Ket in Z basis
def listToKet(ketList,numElts,roll=0):

    if ketList[(-roll)%numElts]<0.1:
        ket = [1,0]#fock(2,1)
    else:
        ket = [0,1]#fock(2,0)

    for i in range(1,numElts):
        if ketList[(i-roll)%numElts]<0.1:
            ket = sparse.kron(ket,[1,0],'csc') #tensor(ket,fock(2,1)) 
        else:
            ket = sparse.kron(ket,[0,1],'csc') #tensor(ket,fock(2,0))

    # ket = ket/ket.norm()

    return ket


# Ket in X basis
def listToXKet(ketList,numElts,roll=0):

    if ketList[(-roll)%numElts]<0.1:
        ket = [1,1]#fock(2,1)
    else:
        ket = [1,-1]#fock(2,0)

    for i in range(1,numElts):
        if ketList[(i-roll)%numElts]<0.1:
            ket = sparse.kron(ket,[1,1],'csc') #tensor(ket,fock(2,1)) 
        else:
            ket = sparse.kron(ket,[1,-1],'csc') #tensor(ket,fock(2,0))

    # ket = ket/ket.norm()

    return ket/2**(L/2)

# returns 1 is word 1 is bigger, -1 if word 2 is bigger, 0 if same
def lexicographicOrder(word1,word2):

    for i in range(len(word1)):

        if word1[i]>word2[i]:
            return 1
        elif word2[i]>word1[i]:
            return -1

    return 0

######### TRANSLATION OPERATOR #########

def find_j(necklace,numElts):

    for i in range(numElts):
        if necklace[numElts-1-i] >0.1:
            return numElts-1-i

    return -1

def theta_funct(necklace,numElts):

    j = find_j(necklace,numElts)

    necklace[j] = 0

    for t in range(1,numElts-j):
        necklace[j+t] = necklace[t-1]

    return int(j)

def findNecklaces(numElts):

    necklaces = []

    necklace = [1]*numElts
    necklaces.append(necklace.copy())

    for i in range(2**numElts):

        j = theta_funct(necklace,numElts)

        if numElts%(j+1)==0:
            necklaces.append(necklace.copy())

        if np.round(np.sum(necklace))==0:
            return necklaces 

# should be lists of 0 or 1's
# returns true if they are the same, else false
def compareStrings(str1,str2):

    length = len(str1)

    for i in range(length):

        if np.abs(str1[i]-str2[i]) > 0.1:
            return False

    return True

# Tells you how many times you can translate the necklace before if it repeats itself
def necklaceCycle(necklace,numElts):

    for i in range(1,numElts):

        # The period must divide the number of sites
        if numElts%i != 0:
            continue 

        if compareStrings(necklace,np.roll(necklace,i)):
            return i

    return numElts


def listToKet(ketList,numElts,roll=0):

    if ketList[(-roll)%numElts]<0.1:
        ket = [1,0]#fock(2,1)
    else:
        ket = [0,1]#fock(2,0)

    for i in range(1,numElts):
        if ketList[(i-roll)%numElts]<0.1:
            ket = sparse.kron(ket,[1,0],'csc') #tensor(ket,fock(2,1)) 
        else:
            ket = sparse.kron(ket,[0,1],'csc') #tensor(ket,fock(2,0))

    # ket = ket/ket.norm()

    return ket


def necklaceToBlochState(necklace,k,numElts,cycle=-1):

    # Check if its all zeros or all ones, need to handle those cases separately.
    # if np.sum(np.abs(np.roll(necklace,1)-necklace))<0.1:
    #     return listToKet(necklace,numElts)

    if cycle==-1:
        cycle = necklaceCycle(necklace,numElts)

    if (cycle*k)%numElts!=0:
        print('Error: Invalid k=',k,' for necklace ',necklace)
        return -1

    ket = listToKet(necklace,numElts)/np.sqrt(cycle)

    for i in range(1,cycle):
        ket += np.exp(2*np.pi*1j*k*i/numElts)*listToKet(necklace,numElts,roll=i)/np.sqrt(cycle)

    # if ket.norm()<0.1:
    #     print('ERROR: zero norm for ',necklace,'with k=',k)
    # else:
    #     norm = ket.norm()
    #     print(norm)
        # ket = ket/norm
        # print(k,necklace,ket.norm())

    return ket

def translation_op(L):
    
    dim=2**L;
    T=np.zeros((dim,dim));
    binaryList=np.zeros((L,L));
    
    for i in range(dim):
        binaryStr=de2bi(i,L);
        T=T+np.dot(np.conj(listToKet(np.roll(binaryStr,1),L)).T,listToKet(binaryStr,L))
    
    return T

######### PARITY OPERATOR #########

def get_P(L):
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    P = sx
    for i in range(L-1):
        P =sparse.kron(sx,P)
    return P


######### PRINT PAULISTRINGS #########

def HS(M1, M2):
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    #M1=np.array(M1)
    #M2=np.array(M2)
    return (M1.conj().T @ M2).trace()


def c2s(c):
    """Return a string representation of a complex number c"""
    if c == 0.0:
        return "0"
    if c.imag == 0:
        return "%g" % np.round(c.real,4)
    elif c.real == 0:
        return "%gj" % np.round(c.imag,4)
    else:
        return "%g+%gj" % (np.round(c.real,4), np.round(c.imag,4))

def decompose2qubit(H):
    """Decompose Hermitian 4x4 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            label = labels[i]  + labels[j]
            a_ij = HS(mkron(S[i], S[j]), H)/2**2
            if np.round(np.abs(a_ij),6) != 0.0:
                print("%s\t *\t (%s)" % (c2s(a_ij), label))
                
def decompose3qubit(H):
    """Decompose Hermitian 8x8 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    count=0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                label = labels[i]  + labels[j] + labels[k]
                a_ij = HS(mkron(S[i], S[j], S[k]), H)/2**3
                if np.round(np.abs(a_ij),6) != 0.0:
                    count=count+1
                    print("%s\t *\t (%s)" % (c2s(a_ij), label))
    print("total of %d non identity strings out of %d" %(count,4**3))
                    
def decompose4qubit(H):
    """Decompose Hermitian 16x16 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    label = labels[i]  + labels[j] + labels[k] + labels[l]
                    a_ij = HS(mkron(S[i], S[j], S[k], S[l]), H)/2**4
                    if np.round(np.abs(a_ij),6) != 0.0:
                        print("%s\t *\t (%s)" % (c2s(a_ij), label))

def decompose5qubit(H):
    """Decompose Hermitian 16x16 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for a in range(4):
                        label = labels[i]  + labels[j] + labels[k] + labels[l] +labels[a]
                        a_ij = HS(mkron(S[i], S[j], S[k], S[l], S[a]), H)/2**5
                        if np.round(np.abs(a_ij),6) != 0.0:
                            print("%s\t *\t (%s)" % (c2s(a_ij), label))
                        
                        
def decompose6qubit(H):
    """Decompose Hermitian 16x16 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for a in range(4):
                        for b in range(4):
                            label = labels[i]  + labels[j] + labels[k] + labels[l] +labels[a]  + labels[b] 
                            a_ij = HS(mkron(S[i], S[j], S[k], S[l], S[a],S[b]), H)/2**6
                            if np.round(np.abs(a_ij),6) != 0.0:
                                print("%s\t *\t (%s)" % (c2s(a_ij), label))
                        
def decompose7qubit(H):
    """Decompose Hermitian 16x16 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for a in range(4):
                        for b in range(4):
                            for c in range(4):
                                label = labels[i]  + labels[j] + labels[k] + labels[l] +labels[a]  + labels[b] + labels[c]
                                a_ij = HS(mkron(S[i], S[j], S[k], S[l], S[a],S[b],S[c]), H)/2**7
                                if np.round(np.abs(a_ij),6) != 0.0:
                                    print("%s\t *\t (%s)" % (c2s(a_ij), label))
    
def decompose8qubit(H):
    """Decompose Hermitian 16x16 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],  [0, -1]])
    id = np.array([[1, 0],  [ 0, 1]])
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    for a in range(4):
                        for b in range(4):
                            for c in range(4):
                                for d in range(4):
                                    label = labels[i]  + labels[j] + labels[k] + labels[l] +labels[a]  + labels[b] + labels[c] + labels[d]
                                    a_ij = HS(mkron(S[i], S[j], S[k], S[l], S[a],S[b],S[c],S[d]), H)/2**8
                                    if np.round(np.abs(a_ij),6) != 0.0:
                                        print("%s\t *\t (%s)" % (c2s(a_ij), label))
            
            
######### DUAL CIRCUITS #########

def dualspace(U):
    Utilde=U
    Utilde=np.reshape(Utilde,[2,2,2,2])
    Utilde=np.transpose(Utilde, [0,2,1,3])
    Utilde=np.reshape(Utilde,[4,4])
    return Utilde


###################################

def shufflestate(psi,vec):
    L=len(vec)
    legs=[int(2) for i in vec]
    psi=np.reshape(psi,legs)
    psi=np.transpose(psi,vec)
    psi=np.reshape(psi,(2**L,1))
    return psi

def shufflevec(A,L):
    vec=list(A)
    for i in range(0,L):
        vec.append(i) if i not in vec else vec
    return vec

def shufflevec_out(A,L):
    vec=list(A)
    for i in range(L,2*L):
        vec.append(i) if i not in vec else vec
    return vec

def Skmean(state,L,k):
    A=list(combinations(list(np.arange(0,L)),k))
    Sfixedk=[]
    for num_comb in range(len(A)):
        vec=shufflevec(A[num_comb],L)
        legs=[int(2) for i in vec]
        psi=state
        psi=np.reshape(psi,legs)
        psi=np.transpose(psi,vec)
        rho=np.outer(psi,psi.conj())
        rho_tensor=np.reshape(rho,legs+legs)
        rho_tensor=np.reshape(rho_tensor,(2**k,2**(L-k),2**k,2**(L-k)))
        rho_reduced=np.trace(rho_tensor,axis1=0,axis2=2)
        S=-np.log(np.trace(rho_reduced@rho_reduced))
        Sfixedk.append(S)
    Skmean=np.mean(Sfixedk)
    return Skmean

def Skmean_rho(rho,L,k):
    A=list(combinations(list(np.arange(0,L)),k))
    Sfixedk=[]
    for num_comb in range(len(A)):
        vecin=shufflevec((A[num_comb]),L)
        Acomp=[x+L for x in A[num_comb]]
        vecout=shufflevec_out(Acomp,L)
        legs=[int(2) for i in vecin+vecout]
        rho=np.array(rho).flatten()
        rho=np.reshape(rho,legs)
        rho=np.transpose(rho,vecin+vecout)
        rho=np.reshape(rho,(2**k,2**(L-k),2**k,2**(L-k)))
        rho_reduced=np.trace(rho,axis1=0,axis2=2)
        S=-np.log(np.trace(rho_reduced@rho_reduced))
        Sfixedk.append(S)
    Skmean=np.mean(Sfixedk)
    return Skmean

def num_ops(n,L):
    return int(comb(L,n)*3**n)

def RedDensity_arbitrarySubsystem(rho,trace_sites,L):
    k=len(trace_sites)
    vecin=shufflevec((trace_sites),L)
    Acomp=[x+L for x in trace_sites]
    vecout=shufflevec_out(Acomp,L)
    legs=[int(2) for i in vecin+vecout]
    rho=np.array(rho).flatten()
    rho=np.reshape(rho,legs)
    rho=np.transpose(rho,vecin+vecout)
    rho=np.reshape(rho,(2**k,2**(L-k),2**k,2**(L-k)))
    rho_reduced=np.trace(rho,axis1=0,axis2=2)
    return rho_reduced


def partial_trace(op,trace_sites,L):
    k=len(trace_sites)
    vecin=shufflevec((trace_sites),L)
    Acomp=[x+L for x in trace_sites]
    vecout=shufflevec_out(Acomp,L)
    legs=[int(2) for i in vecin+vecout]
    op=np.array(op).flatten()
    op=np.reshape(op,legs)
    op=np.transpose(op,vecin+vecout)
    op=np.reshape(op,(2**k,2**(L-k),2**k,2**(L-k)))
    op_reduced=np.trace(op,axis1=0,axis2=2)/2**k
    return op_reduced

def allSubsystems(L):
    trace_sites=[]
    num_trace_sites=[]
    for n in range(0,L+1):
        temp=list(combinations(list(np.arange(0,L)),n))
        trace_sites.append(temp)
        num_trace_sites.append(list(np.ones(len(temp),int)*n))
    trace_sites = [val for sublist in trace_sites for val in sublist]
    num_trace_sites = [val for sublist in num_trace_sites for val in sublist]
    return trace_sites, num_trace_sites

def allSubsystems_andComplement(L):
    R=[]
    Rcomplement=[]
    num_R_sites=[]
    
    for n in range(0,L+1):
        temp=list(combinations(list(np.arange(0,L)),n))
        R.append(temp)
        num_R_sites.append(list(np.ones(len(temp),int)*n))
        #Get complement
        for i in range(len(temp)):
            Rlist = [item.tolist() for item in temp[i]]
            Rlistcomplement=np.arange(0,L)
            Rlistcomplement=Rlistcomplement.tolist()
            for j in range(len(Rlist)):
                Rlistcomplement.remove(Rlist[j])
            Rcomplement.append(Rlistcomplement)
        
    R = [val for sublist in R for val in sublist]
    num_R_sites = [val for sublist in num_R_sites for val in sublist]
        
    return np.array(R), np.array(num_R_sites), np.array(Rcomplement)

#def generating_function_op_nweight(A,L):
#    z = symbols('z', real=False)
#    R, numsitesR, Rcomplement =allSubsystems_andComplement(L)
#    S=0+0*1j
#    for i in range(0,2**L):
#    #Trace out Rcomplement from A, leaves A on R
#        A_onR = RedDensity_arbitrarySubsystem(np.array(A),Rcomplement[i],L)
#        #TrR = np.sum(np.conj(O_onR)*O_onR)
#        TrR = np.trace(A_onR@A_onR)
#        kR=numsitesR[i]
#        print(((1-z)**(L-kR))*((2*z)**kR))
#        S=S+((1-z)**(L-kR))*((2*z)**kR)*TrR
#    return S/2**L
 

def generating_function_op_nweight_zinput(A,L,x):
    R, numsitesR, Rcomplement=allSubsystems_andComplement(L)
    S=0+0*1j
    sarray=np.zeros(2**L,dtype=complex)
    for i in range(0,2**L):
    #Trace out Rcomplement from A, leaves A on R
        #print(['Rcomplement=',Rcomplement[i]])
        kR=numsitesR[i]
        A_onR = RedDensity_arbitrarySubsystem(A,Rcomplement[i],L)
        A_onR = A_onR*(2**kR)
        #TrR = np.sum(np.conj(O_onR)*O_onR)
        TrR = np.trace(A_onR@A_onR)/4**L
        sarray[i]=((1-x)**(L-kR))*((2*x)**kR)
        S=S+((1-x)**(L-kR))*((2*x)**kR)*TrR
    return S/2**L, sarray

def multReduce(iterable):
    return functools.reduce(operator.mul, iterable)

def addReduce(iterable):
    return functools.reduce(operator.add, iterable)

def get_nlweight_direct(s0,x,y,z, Op):
    L = len(x)
    AllTups = []
    for i in range(L):
        AllTups.append([s0[i], x[i], y[i], z[i]])
    AllOpLists = itertools.product(*AllTups)
    AllOps = map(multReduce, list(AllOpLists))
    AllOps = list(AllOps)

    AllTups = []
    for i in range(L):
        AllTups.append(['0', '1', '1', '1'])
    AllOpLists = itertools.product(*AllTups)
    AllOpLabels = map(addReduce, list(AllOpLists))
    AllOpLabels = list(AllOpLabels)
    AllOpNCount = np.array([a.count('1') for a in AllOpLabels])
    AllOplSeparation = np.array([a.rfind('1') -a.find('1') for a in AllOpLabels])
    AllOp_right = np.array([a.rfind('1') for a in AllOpLabels])
    AllOp_left = np.array([a.find('1') for a in AllOpLabels])

    TracesBase = np.zeros(4**L, 'complex')
    for i in range(len(TracesBase)):
        TracesBase[i] = np.trace(np.dot(Op, np.array(AllOps[i].todense())))/(2**L)
    Traces = abs(TracesBase)**2

    AnDirect = [np.sum(Traces[np.where(AllOpNCount==i)]) for i in range(L+1)]
    AlDirect = [np.sum(Traces[np.where(AllOplSeparation==i)]) for i in range(L)]
    ArightDirect = [np.sum(Traces[np.where(AllOp_right==i)]) for i in range(L)]
    AleftDirect = [np.sum(Traces[np.where(AllOp_left==i)]) for i in range(L)]
    return TracesBase, AnDirect, AlDirect, ArightDirect, AleftDirect


def get_op_nweight(A,L,dim=2):
    R, num_R_sites, Rcomplement = allSubsystems_andComplement(L)
    F=np.zeros((2**L,L+1),dtype=complex)
    RedDensityOperatorsTraced=np.zeros(2**L,dtype=complex)
    for i in range(0,2**L):
        kR=num_R_sites[i]
        O_onR = RedDensity_arbitrarySubsystem(np.array(A),Rcomplement[i],L)
        RedDensityOperatorsTraced[i] = np.sum(np.conj(O_onR)*O_onR)# direct product suffices since we're about to trace.
        for m in range(L+1):
            zm=np.exp(1j*m*2*np.pi/(L+1))
            F[i,m]=((1-zm)**(L-kR))*((2*zm)**kR)*RedDensityOperatorsTraced[i]
        ## Discrete FT
    F=np.sum(F,axis=0)/2**L
    Pn=np.zeros(L+1)
    for n in range(L+1):
        for m in range(L+1):
            zmn=np.exp(-1j*m*n*2*np.pi/(L+1))
            Pn[n]=Pn[n]+zmn*F[m]
    return np.abs(np.round(Pn,6))/(L+1)

def kvec(L):
    R, num_R_sites, Rcomplement = allSubsystems_andComplement(L)
    zvec=np.array(np.round([np.exp(-1j*2*np.pi*m/(L+1)) for m in range(L+1)],6))
    kvec=np.zeros((L+1,2**L),dtype=complex)
    for m in range(L+1):
        kvec[m,:]=(1-zvec[m])**(L-num_R_sites)*((2*zvec[m])**num_R_sites)
    return kvec

def get_op_nweight_kvec(A,L,dim=2):
    R, num_R_sites, Rcomplement = allSubsystems_andComplement(L)
    F=np.zeros((2**L,L+1),dtype=complex)
    RedDensityOperatorsTraced=np.zeros(2**L,dtype=complex)
    for i in range(0,2**L):
        kR=num_R_sites[i]
        O_onR = RedDensity_arbitrarySubsystem(np.array(A),Rcomplement[i],L)
        RedDensityOperatorsTraced[i] = np.sum(np.conj(O_onR)*O_onR)# direct product suffices since we're about to trace.
        for m in range(L+1):
            zm=np.exp(1j*m*2*np.pi/(L+1))
            F[i,m]=((1-zm)**(L-kR))*((2*zm)**kR)*RedDensityOperatorsTraced[i]
            
    zvec=np.array(np.round([np.exp(-1j*2*np.pi*m/(L+1)) for m in range(L+1)],6))
    kvec=np.zeros((L+1,2**L),dtype=complex)
    for m in range(L+1):
        kvec[m,:]=(1-zvec[m])**(L-num_R_sites)*((2*zvec[m])**num_R_sites)
    Ak=np.dot(np.conj(kvec),RedDensityOperatorsTraced)/2**L
    ## Discrete FT
    znvec=np.round(np.array([np.array([np.exp(-1j*2*np.pi*m*n/(L+1)) for m in range(L+1)]) for n in range(L+1)]),6)
    Pn=np.dot(Ak,znvec)/(L+1)
    return np.abs(np.round(Pn,6))/(L+1)

def get_correlator(A,B,L):
    return np.round((A@B).diagonal().sum(),6)


def get_correlator_xrange(A,op_list,L):
    Cx=np.zeros(L)
    for x in range(L):
        Cx[x]=np.round((A@op_list[x]).diagonal().sum(),6)
    return Cx


def get_op_lweight(Op, L, dim=2):
    Weight = np.zeros((L,L))
    lNorms = np.zeros((L,L))
    left_norm = np.zeros(L)
    right_norm = np.zeros(L)
    W0 = np.trace(Op)
    for m in range(L):
        for i in range(L-m):
            trace_sites=np.array(np.append(range(0,i), range((i+m+1),L)),int)
            try:
                Weight[m,i] =  OperatorNorm(RedDensity_arbitrarySubsystem(Op, trace_sites, L)/2**len(trace_sites),dim=2)
            except:
                pass
                #print([m,i])
            if m >=2:
                lNorms[m,i] = Weight[m,i] - Weight[m-1,i] - Weight[m-1,i+1]+Weight[m-2, i+1]
            elif m==1:
                lNorms[m,i] = Weight[m,i] - Weight[m-1,i] - Weight[m-1,i+1]+W0
            elif m==0:
                lNorms[m,i] = Weight[m,i] - W0
    right_norm = np.append(Weight[:,0][0],np.diff(Weight[:,0])) 
    left_norm = np.append(Weight[0,L-1], np.diff([Weight[i,L-i-1] for i in range(L)]))[::-1]
    
    return  np.sum(lNorms,axis=1), right_norm, left_norm


######### TRI UNITARY GATES #########

def UCP(phi1,phi2,phi3,h):
    I,X,Y,Z=paulixyz();
    T=translation_op(3);
    VJ_13=T@mkron(I,expm(-1j*(pi/4)*(mkron(X,X)+mkron(Y,Y)+mkron(Z,Z))))@H(T)
    CP12=np.diag([1,1,1,1,1,1,np.exp(1j*phi1),np.exp(1j*phi1)])
    CP23=np.diag([1,1,1,np.exp(1j*phi2),1,1,1,np.exp(1j*phi2)])
    CP31=np.diag([1,1,1,1,1,np.exp(1j*phi3),1,np.exp(1j*phi3)])
    U=mkron(expm(-1j*h[0]*Y),expm(-1j*h[1]*Y),expm(-1j*h[2]*Y))@VJ_13@CP12@CP23@CP31
    return U

def UJtheta(J,theta,h):
    I,X,Y,Z=paulixyz()
    T=translation_op(3)
    VJ_13=T@mkron(I,expm(-1j*(pi/4)*(mkron(X,X)+mkron(Y,Y))-1j*J*mkron(Z,Z)))@H(T)
    Vtheta_2=expm(-1j*theta*mkron(Z,X,Z))
    U=mkron(expm(-1j*h[0]*Y),expm(-1j*h[1]*Y),expm(-1j*h[2]*Y))@VJ_13@Vtheta_2
    return U

def Uperfect():
    s0 = np.eye(2)
    sz = np.diag([1,-1])
    sx = np.array([[0,1], [1,0]])
    ops = np.array([sx,sz,sz,sx,s0])
    gs = []
    for i in range(4):
        gs.append(multikron(np.roll(ops,i,axis=0)))
    gs = [np.kron(s0,g) for g in gs]
    gs.append(multikron([sx]*6))
    gs.append(multikron([sz]*6))
    I = np.eye(64)
    rho = np.copy(I)
    for g in gs:
        rho = np.dot(rho, I+g)/2.
    evals, evecs = np.linalg.eigh(rho)
    psi0 = evecs[:,np.argmax(evals)]

    U = psi0.reshape(8,8) * np.sqrt(2**3)
    return U


def UJtheta_layer(L):
    I,_,_,_=paulixyz();
    if L==8:
        r1=np.random.uniform(0,2*pi,5)
        r2=np.random.uniform(0,2*pi,5)
        
        u1=UJtheta(r1[0],r1[1],r1[2:])
        u2=UJtheta(r2[0],r2[1],r2[2:])
        
        U=mkron(u1,gen_randU(2),u2,gen_randU(2))
        
    elif L==12:
        r1=np.random.uniform(0,2*pi,5)
        r2=np.random.uniform(0,2*pi,5)
        r3=np.random.uniform(0,2*pi,5)
        
        u1=UJtheta(r1[0],r1[1],r1[2:])
        u2=UJtheta(r2[0],r2[1],r2[2:])
        u3=UJtheta(r3[0],r3[1],r3[2:])
        
        U=mkron(u1,gen_randU(2),u2,gen_randU(2),u3,gen_randU(2))
        
    return U

def UJtheta_layer_id(L):
    I,_,_,_=paulixyz();
    if L==8:
        r1=np.random.uniform(0,2*pi,5)
        r2=np.random.uniform(0,2*pi,5)
        
        u1=UJtheta(r1[0],r1[1],r1[2:])
        u2=UJtheta(r2[0],r2[1],r2[2:])
        
        U=mkron(u1,I,u2,I)
        
    elif L==12:
        r1=np.random.uniform(0,2*pi,5)
        r2=np.random.uniform(0,2*pi,5)
        r3=np.random.uniform(0,2*pi,5)
        
        u1=UJtheta(r1[0],r1[1],r1[2:])
        u2=UJtheta(r2[0],r2[1],r2[2:])
        u3=UJtheta(r3[0],r3[1],r3[2:])
        
        U=mkron(u1,I,u2,I,u3,gen_randU(2))
        
    return U

def UCP_layer(L):
    I,_,_,_=paulixyz();
    if L==8:
        r1=np.random.uniform(0,2*pi,6)
        r2=np.random.uniform(0,2*pi,6)
        
        u1=UCP(r1[0],r1[1],r1[2],r1[3:])
        u2=UCP(r2[0],r2[1],r2[2],r2[3:])
        
        U=mkron(u1,gen_randU(2),u2,gen_randU(2))
        
    elif L==12:
        r1=np.random.uniform(0,2*pi,6)
        r2=np.random.uniform(0,2*pi,6)
        r3=np.random.uniform(0,2*pi,6)
        
        u1=UCP(r1[0],r1[1],r1[2],r1[3:])
        u2=UCP(r2[0],r2[1],r2[2],r2[3:])
        u3=UCP(r3[0],r3[1],r3[2],r3[3:])
        
        U=mkron(u1,gen_randU(2),u2,gen_randU(2),u3,gen_randU(2))
                
    return U

def UCP_layer_id(L):
    I,_,_,_=paulixyz();
    if L==8:
        r1=np.random.uniform(0,2*pi,6)
        r2=np.random.uniform(0,2*pi,6)
        
        u1=UCP(r1[0],r1[1],r1[2],r1[3:])
        u2=UCP(r2[0],r2[1],r2[2],r2[3:])
        
        U=mkron(u1,I,u2,I)
        
    elif L==12:
        r1=np.random.uniform(0,2*pi,6)
        r2=np.random.uniform(0,2*pi,6)
        r3=np.random.uniform(0,2*pi,6)
        
        u1=UCP(r1[0],r1[1],r1[2],r1[3:])
        u2=UCP(r2[0],r2[1],r2[2],r2[3:])
        u3=UCP(r3[0],r3[1],r3[2],r3[3:])
        
        U=mkron(u1,I,u2,I,u3,I)
                
    return U
        
        

