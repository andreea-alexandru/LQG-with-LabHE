# Functions that involve quantization (fixed-point arithmetic) and vector operations
import os
import socket
import sys,struct
import json
from gmpy2 import mpz
import paillier
import numpy as np

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False


# When changing default sizes, remember to change in all other classes: Client, Actuator, util_fpv and keys in paillier and LabHE
DEFAULT_KEYSIZE = 1024
DEFAULT_MSGSIZE = 48 
DEFAULT_PRECISION = 24
DEFAULT_SECURITYSIZE = 100
DEFAULT_DGK = 160
NETWORK_DELAY = 0 #0.01 # 10 ms

"""We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
The range N/3 < x < 2N/3 allows for overflow detection."""


def encrypt(pubkey, x, coins=None):
	if (coins==None):
		return pubkey.encrypt(x)
	else: 
		return pubkey.encrypt(x,coins.pop())	

def encrypt_vector(pubkey, x, coins=None):
	size = np.shape(x)
	if len(size) == 1:
		if (coins==None):
			return [pubkey.encrypt(y) for y in x]
		else: 
			return [pubkey.encrypt(y,coins.pop()) for y in x]
	else:
		if (coins==None):
			return pubkey.encrypt(x)
		else: 
			return pubkey.encrypt(x,coins.pop())

def encrypt_matrix(pubkey, x, coins=None):
	size = np.shape(x)
	if len(size) == 2:
		if (coins==None):
			return [[pubkey.encrypt(int(y)) for y in z] for z in x]
		else: return [[pubkey.encrypt(int(y),coins.pop()) for y in z] for z in x]
	else:
		if len(size) == 1:
			if (coins==None):
				return [pubkey.encrypt(y) for y in x]
			else: 
				return [pubkey.encrypt(y,coins.pop()) for y in x]
		else:
			if (coins==None):
				return pubkey.encrypt(x)
			else: 
				return pubkey.encrypt(x,coins.pop())

def encrypt_multi_dim(pubkey, x, dim, coins = None):
	size = len(dim)
	if size > 3:
		if (coins==None):
			return [encrypt_multi_dim(pubkey,y,dim[0:size-1]) for y in x]
		else: return [encrypt_multi_dim(pubkey,y,dim[0:size-1],coins) for y in x]
	else:
		if (coins==None):
			return [encrypt_matrix(pubkey,y) for y in x]
		else: return [encrypt_matrix(pubkey,y,coins) for y in x]

def encrypt_multi_dim_np(pubkey, x, coins = None):
	size = len(x.shape)
	if size > 3:
		if (coins==None):
			return [encrypt_multi_dim_np(pubkey,y) for y in x]
		else: return [encrypt_multi_dim_np(pubkey,y,coins) for y in x]
	else:
		if (coins==None):
			return [encrypt_matrix(pubkey,y) for y in x]
		else: return [encrypt_matrix(pubkey,y,coins) for y in x]


def decrypt_vector(privkey, x):
    return [privkey.decrypt(i) for i in x]

def sum_encrypted_vectors(x, y):
	return [x[i] + y[i] for i in range(np.size(x))]

def diff_encrypted_vectors(x, y):
	return [x[i] - y[i] for i in range(len(x))] 

def mul_sc_encrypted_vectors(x, y): # x is encrypted, y is plaintext
    return [y[i]*x[i] for i in range(len(x))]    

def dot_sc_encrypted_vectors(x, y): # x is encrypted, y is plaintext
    return sum(mul_sc_encrypted_vectors(x,y))

def dot_m_encrypted_vectors(x, A):
    return [dot_sc_encrypted_vectors(x,vec) for vec in A]

def Q_s(scalar,prec=DEFAULT_PRECISION):
	return int(scalar*(2**prec))/(2**prec)

def Q_vector(vec,prec=DEFAULT_PRECISION):
	if np.size(vec)>1:
		return [Q_s(x,prec) for x in vec]
	else:
		return Q_s(vec,prec)

def Q_matrix(mat,prec=DEFAULT_PRECISION):
	return [Q_vector(x,prec) for x in mat]

def fp(scalar,prec=DEFAULT_PRECISION):
	if isinstance(scalar,np.int64):
		scalar = int(scalar)
	if prec < 0:
		prec = int(prec)		# For some reason, without this it throws "conversion error in UI_From_Integer"
		return mpz(gmpy2.t_div_2exp(scalar,-prec))
	else: return mpz(gmpy2.mul(scalar,2**prec))

def fp_vector(vec,prec=DEFAULT_PRECISION):
	if np.size(vec)>1:
		return [fp(x,prec) for x in vec]
	else:
		return fp(vec,prec)

def fp_matrix(mat,prec=DEFAULT_PRECISION):
	return [fp_vector(x,prec) for x in mat]

def retrieve_fp(scalar,prec=DEFAULT_PRECISION):
	return scalar/(2**prec)

def retrieve_fp_vector(vec,prec=DEFAULT_PRECISION):
	return [retrieve_fp(x,prec) for x in vec]

def retrieve_fp_matrix(mat,prec=DEFAULT_PRECISION):
	return [retrieve_fp_vector(x,prec) for x in mat]

def off_gen(pubkey, tx, usk):
	return pubkey.offline_gen_secret(tx,usk)

def off_gen_vec(pubkey, tx, usk):
	size = np.shape(tx)
	if len(size) == 1:
		return [pubkey.offline_gen_secret(y,usk) for y in tx]
	else:
		return pubkey.offline_gen_secret(tx,usk)

def off_gen_mat(pubkey, tx, usk):
	size = np.shape(tx)
	if len(size) == 2:
		return [[pubkey.offline_gen_secret(y,usk) for y in z] for z in tx]
	else:
		if len(size) == 1:
			return [pubkey.offline_gen_secret(y,usk) for y in tx]
		else:
			return pubkey.offline_gen_secret(tx,usk)
	# mat = np.array([[pubkey.offline_gen_secret(y,usk) for y in z] for z in tx])
	# mat = mat.astype(int)
	# return mat

def on_enc(pubkey, x, sx, enc_sx = None):
	if enc_sx is None and not isinstance(sx,paillier.EncryptedNumber):
		return pubkey.encrypt(x,sx)
	else:
		return pubkey.encrypt(x,sx,enc_sx)

def on_enc_vec(pubkey, x, sx, enc_sx = None):
	size = np.shape(x)
	if len(size) == 1:
		if enc_sx is None and not isinstance(sx,paillier.EncryptedNumber):
			return [pubkey.encrypt(x[i],sx[i]) for i in range(size[0])]
		else:
			return [pubkey.encrypt(x[i],sx[i],enc_sx[i]) for i in range(size[0])]
	else:
		return pubkey.encrypt(x,sx,enc_sx)

def on_enc_mat(pubkey, x, sx, enc_sx = None):
	size = np.shape(x)
	print(type(x[0][0]),type(sx[0][0]),type(enc_sx[0][0]))
	if len(size) == 2:
		if enc_sx is None and not isinstance(sx,paillier.EncryptedNumber):
			return [[pubkey.encrypt(x[i][j],sx[i][j]) for j in range(size[1])] for i in range(size[0])]
		else:
			return [[pubkey.encrypt(x[i][j],sx[i][j],enc_sx[i][j]) for j in range(size[1])] for i in range(size[0])]
	else:
		if len(size) == 1:
			return on_enc_vec(pubkey, x, sx, enc_sx)
		else:
			return pubkey.encrypt(x,sx,enc_sx)

def on_dec(privkey, x, sx=None):
	if sx is None:
		return privkey.decrypt(x)
	else:
		return privkey.decrypt(x,sx) 

def on_dec_vec(privkey, x, sx=None):
	if sx is None:
		return [privkey.decrypt(x[i]) for i in range(len(x))]
	else:
		return [privkey.decrypt(x[i],sx[i]) for i in range(len(x))]

def on_dec_mat(privkey, x, sx=None):
	if sx is None:
		return [[privkey.decrypt(x[i][j]) for j in range(len(x[0]))] for i in range(len(x))]
	else:
		return [[privkey.decrypt(x[i][j],sx[i][j]) for j in range(len(x[0]))] for i in range(len(x))]

### Vectorize returns everything as np.ndarray

vfp = np.vectorize(fp)
vretrieve_fp = np.vectorize(retrieve_fp)
voff_gen = np.vectorize(off_gen)
vencrypt = np.vectorize(encrypt)
von_enc = np.vectorize(on_enc)
von_dec = np.vectorize(on_dec)