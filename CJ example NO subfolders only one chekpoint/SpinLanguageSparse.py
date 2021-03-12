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

# Takes in a binary number b stored as an array.
def bin2dec(b):
    b= np.array(b)[::-1]
    d=0
    for i in range(b.shape[0]):
       d=d+b[i]*2**i
    return d

def arr2baseb(arr, b):
    arr= np.array(arr)[::-1]
    d=0
    for i in range(arr.shape[0]):
       d=d+arr[i]*b**i
    return d

def dec2bin(d,n):
    a =  [0]*(n - len(list(bin(d)[2:]))) + list(bin(d)[2:])
    return np.array(a,dtype=int)

def powerset(iterable):
  xs = list(iterable)
  # note we return an iterator rather than a list
  return list(chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) ))

def revbits(x,L):
    return int(bin(x)[2:].zfill(L)[::-1], 2)

def flipbits(x,L):
    dim=2**L
    return dim-x-1

def int2baseb(x, base, L, ret=False):
    digs = string.digits + string.letters
    if x < 0: sign = -1
    elif x == 0:
        if ret:
            return np.array([0]*L), '0'*L

        return np.array([0]*L)#'0'*L
    else: sign = 1
    x *= sign
    digits = []
    while x:
        digits.append(digs[x % base])
        x /= base
    if sign < 0:
        digits.append('-')
    digits.reverse()
    digstring = '0'*(L-len(digits))+''.join(digits)
    digits =  np.array(digits, int)
    if ret:
        return np.append([0]*(L-len(digits)), digits), digstring
    return np.append([0]*(L-len(digits)), digits)#, digstring


def rotateleft(s,d): 
    # slice string in two parts for left and right 
    Rfirst = s[0 : len(s)-d] 
    Rsecond = s[len(s)-d : ] 
  
    return (Lsecond + Lfirst) 
    print "Right Rotation : ", (Rsecond + Rfirst) 

def rotateright(s,d): 
    # slice string in two parts for left and right 
    Rfirst = s[0 : len(s)-d] 
    Rsecond = s[len(s)-d : ] 
  
    return (Rsecond + Rfirst) 

def transbits(i, q, L):
    _,s = int2baseb(i,q,L, ret=True)
    sp = rotateright(s, 1)
    return int(sp, q)
    

def gen_translation(L, q):
    transmat = sparse.lil_matrix((q**L, q**L))
    for i in range(q**L):
        ip = transbits(i, q, L)
        transmat[ip, i]=1
    return sparse.csr_matrix(transmat)




def gen_reflection(L):
    reflectMat = sparse.lil_matrix((2**L, 2**L))
    for i in range(2**L):
        reflectMat[revbits(i,L),i]=1
    return sparse.csr_matrix(reflectMat)
        
def gen_U1Proj(symVec):
    vals =np.unique(symVec)
    PList=[]
    for v in vals:
        ind = np.where(symVec==float(v))
        dim = np.size(ind)
        P = sparse.lil_matrix((dim,len(symVec)))
        for j in range(dim):
            P[j,ind[0][j]] = 1.0
        PList.append(P)
    return PList

def gen_parityProj_xbasis(L):
    dim = 2**L
    Pxp = sparse.lil_matrix((dim/2,dim ))
    Pxm = sparse.lil_matrix((dim/2,dim ))
    for j in range(dim/2):
        Pxp[j,j]=1.0/np.sqrt(2)
        Pxp[j, dim-j-1] = 1.0/np.sqrt(2)
        Pxm[j,j]=1.0/np.sqrt(2)
        Pxm[j, dim-j-1] = -1.0/np.sqrt(2)
    return Pxp, Pxm

def gen_reflectionProj(L):
    dim = 2**L

    dmin={}
    dequal={}
    for i in range(2**L):
        j=revbits(i,L)
        if j!=i:
            dmin[np.min([i,j])]=np.max([i,j])
        else:
            dequal[i]=i
            
    keys = dmin.keys()
    keysEqual = dequal.keys()

    dimm= len(keys)
    dimp = dimm + len(keysEqual)
    
    Pm = sparse.lil_matrix((dimm,dim))
    Pp = sparse.lil_matrix((dimp,dim))

    for i in range(dimm):
        k=keys[i]
        Pp[i, k] = 1.0/np.sqrt(2)
        Pp[i, dmin[k]]=1.0/np.sqrt(2)
        Pm[i, k] = 1.0/np.sqrt(2)
        Pm[i, dmin[k]]=-1.0/np.sqrt(2)

    for i in range(2**(L/2)):
        k=keysEqual[i]
        Pp[dimm+i, k]=1.0    

    return sparse.csr_matrix(Pp), sparse.csr_matrix(Pm)



def gen_reflectionProj_U1Sector(L, inds):
    dim = len(inds)
    dmin={}
    dequal={}
    for i in inds:
        j=revbits(i,L)
        iind = np.where(inds==i)[0][0]
        jind = np.where(inds==j)[0][0]
        if j!=i:
            dmin[np.min([iind,jind])]=np.max([iind,jind])
        else:
            dequal[iind]=iind
            
    keys = dmin.keys()
    keysEqual = dequal.keys()

    dimm= len(keys)
    dimp = dimm + len(keysEqual)
    
    Pm = sparse.lil_matrix((dimm,dim))
    Pp = sparse.lil_matrix((dimp,dim))

    for i in range(dimm):
        k=keys[i]
        Pp[i, k] = 1.0/np.sqrt(2)
        Pp[i, dmin[k]]=1.0/np.sqrt(2)
        Pm[i, k] = 1.0/np.sqrt(2)
        Pm[i, dmin[k]]=-1.0/np.sqrt(2)

    for i in range(len(keysEqual)):
        k=keysEqual[i]
        Pp[dimm+i, k]=1.0    

    return sparse.csr_matrix(Pp), sparse.csr_matrix(Pm)

def gen_parityAndReflectionProj(L):
    dim = 2**L

    dimmp = (dim - 2*(2**(L/2)))/4
    dimpp = dimmp + (2**(L/2))/2 + (2**(L/2))/2
    dimpm = dimmp + 2**(L/2)/2
    dimmm = dimmp+ 2**(L/2)/2

    # Ppm = projector onto states with +1 eigenvalue for reflection and -1 for parity etc.
    Ppp = sparse.lil_matrix((dimpp,dim))
    Ppm = sparse.lil_matrix((dimpm,dim))
    Pmp = sparse.lil_matrix((dimmp,dim))
    Pmm = sparse.lil_matrix((dimmm,dim))

    d={}; dRevEqual={}; dParRevEqual={}
    for i in range(2**L):
        j=revbits(i,L); ifl= dim-i-1; jfl=dim-j-1;
        orbit = [i,j, ifl, jfl]
        if i==j:
            dRevEqual[np.min(orbit)] = np.unique(orbit)
        elif j==ifl:
            dParRevEqual[np.min(orbit)] = np.unique(orbit)
        else:
            d[np.min(orbit)]= orbit
            
    k=d.keys(); kRevEqual=dRevEqual.keys(); kParRevEqual=dParRevEqual.keys()
    for i in range(dimmp):
        orbit = d[k[i]]
        for o in range(len(orbit)):
            Ppp[i,orbit[o]] = 0.5
            Ppm[i,orbit[o]] = ((-1)**(o/2))*0.5
            Pmp[i,orbit[o]] = ((-1)**(o))*0.5
            Pmm[i,orbit[o]] = ((-1)**((o+1)/2))*0.5 
        
    for i in range((2**(L/2))/2):
        orbit = dParRevEqual[kParRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+i,orbit[o]] = 1/np.sqrt(2)
            Pmm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            

    for i in range((2**(L/2))/2):
        orbit = dRevEqual[kRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+(2**(L/2))/2+i,orbit[o]] = 1/np.sqrt(2)
            Ppm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            
    return sparse.csr_matrix(Ppp), sparse.csr_matrix(Ppm), sparse.csr_matrix(Pmp), sparse.csr_matrix(Pmm)
        
def gen_parityAndReflectionProj_U1Sector(L, inds):
    dim = len(inds)

    d={}; dRevEqual={}; dParRevEqual={}
    for i in inds:
        j=revbits(i,L); ifl= 2**L-i-1; jfl=2**L-j-1;
        orbit = [np.where(inds==i)[0][0],np.where(inds==j)[0][0], np.where(inds==ifl)[0][0], np.where(inds==jfl)[0][0]]
        if i==j:
            dRevEqual[np.min(orbit)] = np.unique(orbit)
        elif j==ifl:
            dParRevEqual[np.min(orbit)] = np.unique(orbit)
        else:
            d[np.min(orbit)]= orbit

    k=d.keys(); kRevEqual=dRevEqual.keys(); kParRevEqual=dParRevEqual.keys()

    dimmp = len(k)
    dimpp = dimmp + len(kRevEqual) + len(kParRevEqual)
    dimpm = dimmp + len(kRevEqual)
    dimmm = dimmp+ len(kParRevEqual)

    # Ppm = projector onto states with +1 eigenvalue for reflection and -1 for parity etc.
    Ppp = sparse.lil_matrix((dimpp,dim))
    Ppm = sparse.lil_matrix((dimpm,dim))
    Pmp = sparse.lil_matrix((dimmp,dim))
    Pmm = sparse.lil_matrix((dimmm,dim))

    for i in range(dimmp):
        orbit = d[k[i]]
        for o in range(len(orbit)):
            Ppp[i,orbit[o]] = 0.5
            Ppm[i,orbit[o]] = ((-1)**(o/2))*0.5
            Pmp[i,orbit[o]] = ((-1)**(o))*0.5
            Pmm[i,orbit[o]] = ((-1)**((o+1)/2))*0.5 
        
    for i in range(len(kParRevEqual)):
        orbit = dParRevEqual[kParRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+i,orbit[o]] = 1/np.sqrt(2)
            Pmm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            

    for i in range(len(kRevEqual)):
        orbit = dRevEqual[kRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+len(kParRevEqual)+i,orbit[o]] = 1/np.sqrt(2)
            Ppm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            
    return sparse.csr_matrix(Ppp), sparse.csr_matrix(Ppm), sparse.csr_matrix(Pmp), sparse.csr_matrix(Pmm)
        



def diagonalizeWithSymmetries(H, PList,L):
    ind = 0
    Allevecs = np.zeros((2**L, 2**L), 'complex')
    Allevals = np.zeros((2**L))
    for P in PList:
        H_sym = P*H*(P.T)
        if (H_sym - np.conj(H_sym)).size ==0:
            H_sym = np.real(np.array(H_sym.todense()))
        else:
            H_sym = np.array(H_sym.todense())

        evals, evecs= linalg.eigh(H_sym)
        Allevecs[:, ind:ind+len(evecs)] = np.dot(np.array((P.T).todense()), evecs)
        Allevals[ind:ind+len(evecs)] = evals
        ind = ind+len(evecs)
    ind = np.argsort(Allevals)
    Allevals=Allevals[ind]
    Allevecs=Allevecs[:,ind]

    return Allevals, Allevecs

def diagonalizeUnitaryWithSymmetries(U, PList,L):
    ind = 0
    Allevecs = np.zeros((PList[0].shape[1], PList[0].shape[1]),'complex')
    Allevals = np.zeros((PList[0].shape[1]),'complex')

    for P in PList:
        Pdense = np.array(P.todense())
        PTdense = np.array((P.T).todense())
        U_sym = np.dot(np.dot(Pdense, U), PTdense)
        evals, evecs= linalg.eig(U_sym)
        Allevecs[:, ind:ind+len(evecs)] = np.dot(PTdense, evecs)
        Allevals[ind:ind+len(evecs)] = evals
        ind = ind+len(evecs)

    return Allevals, Allevecs

# Because of Sz definition, assumes a basis ordering like 11, 10, 01, 00 for 2 spins 
def gen_spmxyz(L): 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    sp = sparse.csr_matrix([[0., 2.],[0, 0.]])/np.sqrt(2)
    sm = sparse.csr_matrix([[0., 0],[2., 0.]])/np.sqrt(2)

    sz_list = []
    sp_list = [] 
    sm_list = []
    s0_list =[]
    I = sparse.csr_matrix(np.eye(2**L))
    
    for i_site in range(L): 
        if i_site==0: 
            Z=sz
            P=sp
            M=sm
        else: 
            Z= sparse.csr_matrix(np.eye(2))
            P= sparse.csr_matrix(np.eye(2))
            M= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                Z=sparse.kron(Z,sz, 'csr')
                P=sparse.kron(P,sp, 'csr')
                M=sparse.kron(M,sm, 'csr')
            else: 
                Z=sparse.kron(Z,np.eye(2),'csr') 
                P=sparse.kron(P,np.eye(2), 'csr')
                M=sparse.kron(M,np.eye(2), 'csr')

        sz_list.append(Z) 
        sp_list.append(P) 
        sm_list.append(M) 
        s0_list.append(I)
    return s0_list, sp_list, sm_list, sz_list


def gen_upmd(L): 
    su = sparse.csr_matrix([[2., 0],[0, 0.]])/np.sqrt(2)
    sd = sparse.csr_matrix([[0., 0],[0, 2.]])/np.sqrt(2)
    sp = sparse.csr_matrix([[0., 2.],[0, 0.]])/np.sqrt(2)
    sm = sparse.csr_matrix([[0., 0],[2., 0.]])/np.sqrt(2)

    su_list = []
    sp_list = [] 
    sm_list = []
    sd_list =[]
    
    for i_site in range(L): 
        if i_site==0: 
            U=su
            P=sp
            M=sm
            D=sd
        else: 
            U= sparse.csr_matrix(np.eye(2))
            D= sparse.csr_matrix(np.eye(2))
            P= sparse.csr_matrix(np.eye(2))
            M= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                U=sparse.kron(U,su, 'csr')
                D=sparse.kron(D,sd, 'csr')
                P=sparse.kron(P,sp, 'csr')
                M=sparse.kron(M,sm, 'csr')
            else: 
                U=sparse.kron(U,np.eye(2),'csr') 
                D=sparse.kron(D,np.eye(2),'csr') 
                P=sparse.kron(P,np.eye(2), 'csr')
                M=sparse.kron(M,np.eye(2), 'csr')

        su_list.append(U) 
        sp_list.append(P) 
        sm_list.append(M) 
        sd_list.append(D)
    return su_list, sp_list, sm_list, sd_list



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



def gen_pud_list(L): 
    su = sparse.csr_matrix([[1., 0],[0, 0.]])
    sd = sparse.csr_matrix([[0., 0],[0, 1.]])

    su_list = []
    sd_list =[]
    
    for i_site in range(L): 
        if i_site==0: 
            U=su
            D=sd
        else: 
            U= sparse.csr_matrix(np.eye(2))
            D= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                U=sparse.kron(U,su, 'csr')
                D=sparse.kron(D,sd, 'csr')
            else: 
                U=sparse.kron(U,np.eye(2),'csr') 
                D=sparse.kron(D,np.eye(2),'csr') 

        su_list.append(U) 
        sd_list.append(D) 
    return su_list, sd_list
 

def gen_randU(n):
    X = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2);
    [Q,R] = np.linalg.qr(X);
    R = np.diag(np.diag(R)/abs(np.diag(R)));
    return np.dot(Q,R)

def gen_randvec(n):
    X = (np.random.randn(n) + 1j*np.random.randn(n))/np.sqrt(2);
    return X/linalg.norm(X)
    


def gen_translated_gate(U, L):
    k = int(np.log2(U.shape[0]))
    gatelist = []
    s0 = sparse.csr_matrix(np.eye(2)) 

    for i_site in range(L-k+1):
        gate = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), U) , sparse.csr_matrix(np.eye(2**(L-i_site-k))))
        gatelist.append(sparse.csr_matrix(gate))

    return gatelist


def gen_randgate_list(l, L):
    gatelist = []
    s0 = sparse.csr_matrix(np.eye(2)) 

    for i_site in range(L-l+1):
        U = gen_randU(2**l)
        gate = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), U) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
        gatelist.append(sparse.csr_matrix(gate))
    return gatelist



def gen_unifgate_list_U1(L, Tr =[], bc='obc'):
    gatelist = []
    s0 = sparse.csr_matrix(np.eye(2))
    l=2
    U = np.zeros((2**2, 2**2), 'complex')
    U[0,0] = gen_randU(1)
    U[1:3, 1:3] = gen_randU(2)
    U[3,3] = gen_randU(1)

    for i_site in range(L-1):
        gate = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), U) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
        gatelist.append(sparse.csr_matrix(gate))

    if bc=='pbc':
        gate = sparse.kron(sparse.csr_matrix(np.eye(2**(L-2))), U)
        gate =  sparse.csr_matrix((Tr)*(gate)*(Tr.T))        
        gatelist.append(sparse.csr_matrix(gate))
    
    return gatelist

def gen_randgate_list_U1(L, Tr =[], bc='obc'):
    gatelist = []
    s0 = sparse.csr_matrix(np.eye(2))
    l=2

    for i_site in range(L-1):
        U = np.zeros((2**2, 2**2), 'complex')
        U[0,0] = gen_randU(1)
        U[1:3, 1:3] = gen_randU(2)
        U[3,3] = gen_randU(1)
        gate = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), U) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
        gatelist.append(sparse.csr_matrix(gate))

    if bc=='pbc':
        
        U = np.zeros((2**2, 2**2), 'complex')
        U[0,0] = gen_randU(1)
        U[1:3, 1:3] = gen_randU(2)
        U[3,3] = gen_randU(1)
        gate = sparse.kron(sparse.csr_matrix(np.eye(2**(L-2))), U)
        gate =  sparse.csr_matrix((Tr)*(gate)*(Tr.T))        
        gatelist.append(sparse.csr_matrix(gate))
    
    return gatelist


def gen_randgate_list_U1_noninteracting(L, Tr =[], bc='obc'):
    gatelist = []
    s0 = sparse.csr_matrix(np.eye(2))
    l=2

    for i_site in range(L-1):
        U = np.zeros((2**2, 2**2), 'complex')
        U[0,0] = 1
        U[1:3, 1:3] = gen_randU(2)
        d = np.linalg.det(U[1:3, 1:3])        
        U[3,3] = d**2
        U = gen_randU(1)[0][0]*U
        gate = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), U) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
        gatelist.append(sparse.csr_matrix(gate))

    if bc=='pbc':
        
        U = np.zeros((2**2, 2**2), 'complex')
        U[0,0] = 1. 
        U[1:3, 1:3] = gen_randU(2)
        d = np.linalg.det(U[1:3, 1:3])        
        U[3,3] = d**2

        U = gen_randU(1)[0][0]*U
        
        gate = sparse.kron(sparse.csr_matrix(np.eye(2**(L-2))), U)
        gate =  sparse.csr_matrix((Tr)*(gate)*(Tr.T))        
        gatelist.append(sparse.csr_matrix(gate))
    
    return gatelist





##def gen_projlist(l, L):
##    projlist = []
##    s0 = sparse.csr_matrix(np.eye(2))
##
##    for val in range(2**l):
##        projlistval = []
##        P = sparse.csr_matrix((2**l, 2**l))
##        P[val, val]=1
##        for i_site in range(L-l+1):
##            proj = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), P) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
##            projlistval.append(proj)
##
##        projlist.append(projlistval)
##
##    return projlist


def fwht(a):
    """In-place Fast Walsh-Hadamard Transform of array a"""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
        h *= 2



def gen_onsiteprojlist_xbasis( L):
    projlist = []

    Pright = np.array([[1,1],[1,1]])
    Pleft = np.array([[1,-1],[-1,1]])
    Pboth = [Pright, Pleft]
    
    
    for i_site in range(L):
        projlistval = []
        for val in range(2):
            P=sparse.csr_matrix(Pboth[val])
            proj = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), P) , sparse.csr_matrix(np.eye(2**(L-i_site-1))))
            projlistval.append(sparse.csr_matrix(proj))

        projlist.append(projlistval)

    return projlist


def gen_projlist_all(l, L):
    projlist = []
    s0 = sparse.csr_matrix(np.eye(2))

    for i_site in range(L-l+1):
        projlistval = []
        for val in range(2**l):
            P = sparse.lil_matrix((2**l, 2**l))
            P[val, val]=1
            P=sparse.csr_matrix(P)
            proj = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), P) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
            projlistval.append(sparse.csr_matrix(proj))

        projlist.append(projlistval)

    return projlist
    

def gen_rand_projlist(l,L):
    projlist = []
    s0 = sparse.csr_matrix(np.eye(2))

    for i_site in range(L-l+1):
        val = np.random.randint(2**l)
        P = sparse.csr_matrix((2**l, 2**l))
        P[val, val]=1
        proj = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), P) , sparse.csr_matrix(np.eye(2**(L-i_site-l))))
        projlist.append(proj)

    return projlist        


# efficient way to generate Sz and Sx operator lists lists without doing sparse matrix multiplications
def gen_sxz_fast(L):
    sz_list = []
    sx_list = []
    rows = np.arange(2**L)
    data = np.ones(2**L)

    for i in range(L-1,-1,-1):
        unit =[1]*(2**i)+[-1]*(2**i)
        zvec = np.array(unit*(2**(L-i-1)), int)

        cols = rows + (2**i)*zvec
        
        sz_list.append(scipy.sparse.diags(zvec, dtype = int))
        sx_list.append(scipy.sparse.csc_matrix((data,(rows, cols)), shape = (2**L, 2**L), dtype = int))
                       
    return sx_list,sz_list


def gen_onsite_projlist_Pauli(opList):
    L = len(opList)
    if L <1:
        return []

    projlist = []
    eye = scipy.sparse.eye(opList[0].size)

    for i in range(L):
        projlist.append([eye+opList[i], eye-opList[i]])
    return projlist
        
def gen_onsite_U_Pauli(opList, thetaList):
    L = len(opList)
    if L <1:
        return []

    UList = []
    eye = scipy.sparse.eye(opList[0].size)

    for i in range(L):
        UList.append(np.cos(thetaList[i])*eye- 1j*np.sin(thetaList[i])*opList[i])
    return UList


def gen_kdist_prods(op_list, op_list2=[],k=1, bc='obc'):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list


    opprods=[]
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        opprods.append(op_list[i]*op_list2[np.mod(i+k,L)])
    return opprods



def gen_onsiteprojlist_xbasis( L):
    projlist = []

    Pright = np.array([[1,1],[1,1]])
    Pleft = np.array([[1,-1],[-1,1]])
    Pboth = [Pright, Pleft]
    
    
    for i_site in range(L):
        projlistval = []
        for val in range(2):
            P=sparse.csr_matrix(Pboth[val])
            proj = sparse.kron(sparse.kron(sparse.csr_matrix(np.eye(2**i_site)), P) , sparse.csr_matrix(np.eye(2**(L-i_site-1))))
            projlistval.append(sparse.csr_matrix(proj))

        projlist.append(projlistval)

    return projlist



def gen_op_total(op_list):
    L = len(op_list)
    tot = op_list[0]
    for i in range(1,L): 
        tot = tot + op_list[i] 
    return tot

def gen_op_prod(op_list):
    L= len(op_list)
    P = op_list[0]
    for i in range(1, L):
        P = P*op_list[i]
    return P

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

    
def gen_lr_int(op_list, alpha, Jlist = [], bc='obc'):
    L= len(op_list)
    interaction=0*op_list[0]
    if Jlist == []:
        Jlist = np.ones((L, L))
    for i in range(L):
        for j in range(i+1,L):
            r=1.0*abs(i-j)
            if bc=='pbc':
                r=np.min([r, L-r])
            interaction = interaction+ Jlist[i,j]*(op_list[i]*op_list[j])/(r**alpha)
    return interaction

def gen_lr_exp_int(op_list, xi, Jlist = [], bc='obc'):
    L= len(op_list)
    interaction=0*op_list[0]
    if Jlist == []:
        Jlist = np.ones((L, L))
    for i in range(L):
        for j in range(i+1,L):
            r=1.0*abs(i-j)
            if bc=='pbc':
                r=np.min([r, L-r])
            interaction = interaction+ Jlist[i,j]*(op_list[i]*op_list[j])*np.exp(-r/xi)
    return interaction
 
def gen_lr_exp_int_trunc(op_list, xi, Jlist, trunc, bc='obc'):
    L= len(op_list)
    interaction=0*op_list[0]
    if Jlist == []:
        Jlist = np.ones((L, L))
    for i in range(L):
        for j in range(i+1,L):
            r=1.0*abs(i-j)
            if bc=='pbc':
                r=np.min([r, L-r])
            if r <= trunc:
                interaction = interaction+ Jlist[i,j]*(op_list[i]*op_list[j])*np.exp(-r/xi)
    return interaction




def gen_lr_int_trunc_Jmat(L, Jmean, Jvar, trunc):
    nfloor = np.floor(trunc)
    nceil = np.ceil(trunc)
    
    p = nceil - trunc

    JList = np.zeros((L,L))
    for i in range(L):
        for j in range(i+1,L):
            r=j-i

            if r <= nfloor:
                JList[i,j] =np.random.uniform(Jmean-Jvar, Jmean+Jvar)

            if r == nceil:
                bv = np.random.uniform(0,1) < 1-p
                if bv:
                    JList[i,j] = np.random.uniform(Jmean-Jvar, Jmean+Jvar)
    return JList


# generates \sum_i O_i O_{i+k} type interactions with strength J_{i,i+k}
def gen_interaction_Jmat(op_list, J_list, op_list2=[]):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    H = sparse.csr_matrix(op_list[0].shape)
    for i in range(L):
        for j in range(L):
            H = H+ J_list[i,j]*op_list[i]*op_list2[j]
    return H




def gen_diag_projector(symMatrix, symValue):
    symMatrix = symMatrix.diagonal()
    ind = np.where(symMatrix==symValue)
    dim = len(symMatrix)
    dim0 = np.size(ind)
    P = sparse.lil_matrix((dim0,dim ))
    for i in range(np.size(ind)):
        P[i,ind[0][i]] = 1.0
    return P


def projectedStateNum_ToState(symMatrix, symValue, psnum):
    symMatrixDense = symMatrix.todense()
    ind = np.where(np.diag(symMatrixDense)==symValue)
    dimOrig = len(symMatrixDense)
    
    return dec2bin(dimOrig - 1 -ind[0][psnum], int(np.log2(dimOrig)))
    
    
def gen_state_bloch(thetaList, phiList):
    L=len(thetaList)
    psi = np.kron([np.cos(thetaList[0]/2.),np.exp(1j*phiList[0])*np.sin(thetaList[0]/2.)],
                  [np.cos(thetaList[1]/2.),np.exp(1j*phiList[1])*np.sin(thetaList[1]/2.)])
    for i in range(2,L):
        psi = np.kron(psi, [np.cos(thetaList[i]/2.),np.exp(1j*phiList[i])*np.sin(thetaList[i]/2.)])
    return psi

def isdiag(M):
    if M.ndim!=2:
        return False
    return np.all(M == np.diag(np.diag(M)))

def gen_unifstate_bloch(theta, phi,L):
    return gen_state_bloch([theta]*L, [phi]*L)

def gen_U(H_sparse):
    H = np.array(H_sparse.todense())
    if isdiag(H):
        return np.diag(np.exp(-1j*np.diag(H)))
    return linalg.expm(-1j*H)

def gen_diagonalEnsemble(psi0, evecs,op):
    psi0Init = np.dot(np.conj(evecs.T), psi0)
    dim= len(evecs)
    OPMat = gen_diagonal_ME(op,evecs)
    return np.dot(OPMat, abs(psi0Init)**2)



def LevelStatistics(energySpec, ret=False):
    delta = energySpec[1:] -energySpec[0:-1]
    delta = abs(delta[abs(delta) > 10**-12])
    r = map(lambda x,y: min(x,y)*1.0/max(x,y), delta[1:], delta[0:-1])
    if ret==True:
        return np.array(r), np.mean(r)
    return np.mean(r)

"""returns np.dot(A,B) with speedups for diagonal matrices. """
def mydot(A, B):
    if isdiag(A):
        if isdiag(B):
            return A*B
        return (np.diag(A)*(B.T)).T
    if isdiag(B):
        return A*np.diag(B)
    else:
        return np.dot(A,B)


##def gen_state_bloch(theta, phi,L):
##    psi = np.kron([np.cos(theta/2.),np.exp(1j*phi)*np.sin(theta/2.)],[np.cos(theta/2.),np.exp(1j*phi)*np.sin(theta/2.)])
##    for i in range(2,L):
##        psi = np.kron(psi, [np.cos(theta/2.),np.exp(1j*phi)*np.sin(theta/2.)])
##    return psi
        
def ME(op, state):
    return mydot(mydot(np.conj(state), op),state)

"returns a list \langle \alpha |O|\alpha \rangle for all eigenvectors \alpha"
def gen_diagonal_ME(op, evecs):
    return np.sum((np.conj(evecs))*mydot(op, evecs),0)

def gen_allME(op, evecs):
    return mydot(mydot(np.conj(evecs.T),op), evecs)

def timeEvolutionStrobos(evals, evecs, psi0, OpList, nPeriods):
    nops = len(OpList)
    Times=np.arange(0,nPeriods)
    freq = 2*np.pi*np.fft.fftfreq(Times.shape[-1]);freq = freq+2*np.pi*(freq<0)

    OpTimesList = np.zeros((nops, nPeriods))
    OpFourierList = np.zeros((nops, nPeriods),'complex')

    psi0Init = np.dot(np.conj(evecs.T), psi0)

    OpMatList = np.zeros(nops, 'object')
    for i in range(nops):
        OpMatList[i] = gen_allME(OpList[i], evecs)

    for i in range(len(Times)):
        t=Times[i]
        psi = psi0Init*np.exp(-1j*evals*t)
        for j in range(nops):
            OpTimesList[j, i] = ME(OpMatList[j], psi)

    for j in range(nops):
        OpFourierList[j] = np.fft.fft(OpTimesList[j])

    return OpTimesList, OpFourierList


def timeEvolution(evals, evecs, psi0, OpList, Times):
    nops = len(OpList)
    OpTimesList = np.zeros((nops, len(Times)))
    psi0Init = np.dot(np.conj(evecs.T), psi0)

    OpMatList = np.zeros(nops, 'object')
    for i in range(nops):
        OpMatList[i] = gen_allME(OpList[i].toarray(), evecs)

    for i in range(len(Times)):
        t=Times[i]
        psi = psi0Init*np.exp(-1j*evals*t)
        for j in range(nops):
            OpTimesList[j, i] = ME(OpMatList[j], psi)

    return OpTimesList



def timeEvolution_allinitzstates(evals, evecs, OpList, Times):
    nops = len(OpList)
    dim = len(evals)
    OpTimesList = np.zeros((dim, nops, len(Times)))
    psi0Init = np.conj(evecs.T)

    if abs(Times[0])>10**-12:
        Times = np.append([0.], Times)

    OpMatList = np.zeros(nops, 'object')
    for i in range(nops):
        OpMatList[i] = gen_allME(OpList[i].toarray(), evecs)


    for j in range(nops):
        OpTimesList[:,j, 0] = gen_diagonal_ME(OpMatList[j], psi0Init)
    

    for i in range(1,len(Times)):
        t=Times[i]
        psi = (np.exp(-1j*evals*t)*psi0Init.T).T
        for j in range(nops):
            OpTimesList[:,j, i] = gen_diagonal_ME(OpMatList[j], psi)*OpTimesList[:,j, 0]

    OpTimesList[:,:,0] = OpTimesList[:,:,0]**2

    return OpTimesList

# converts the eigenvalues of U(T) -- phases -- to quasienergies       
def phaseToQuasienergy(evals, evecs=[]):
    evals = (np.real(1j*np.log(evals)))
    indSort = np.argsort(evals)
    evals = evals[indSort]
    if evecs == []:
        return evals
    evecs = evecs[:, indSort]
    return evals, evecs

    
