#!/usr/bin/env python3

from gmpy2 import mpz
import paillier
import numpy as np
import time
import os
import LabHE
import util_fpv

DEFAULT_KEYSIZE = 1024 
DEFAULT_SEEDSIZE = 32
DEFAULT_MSGSIZE = 48 
DEFAULT_PRECISION = 24
DEFAULT_SECURITYSIZE = 100
NETWORK_DELAY = 0 #0.01 # 10 ms

"""We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
The range N/3 < x < 2N/3 allows for overflow detection. The product of the hashes has to fit in the 
message space, products of secrets do not exceed N/3!"""

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

class Actuator:
	def __init__(self, usk_setup, usk_client, key_length=DEFAULT_KEYSIZE):
		"""Need one usk for each client, and a usk for the actuator, for 
		assigning the labels for the refreshed values. For simplicity, in 
		this implementation, we have one client representing the setup and 
		one client representing the sensors."""
		self.usk = usk_setup,usk_client
		self.usk_a = gmpy2.mpz_urandomb(gmpy2.random_state(),DEFAULT_SEEDSIZE)
		pubkey, privkey = LabHE.generate_LabHE_keypair(self.usk, key_length)
		msk = privkey.msk; upk = privkey.upk
		self.msk = msk
		self.mpk = pubkey.Pai_key
		self.upk = upk
		self.privkey = privkey
		self.pubkey = pubkey

	def genLabels(self,n,m,p,T,flag):
		self.n = n; self.m = m; self.p = p; self.T = T
		lf = DEFAULT_PRECISION
		usk = self.usk
		usk_a = self.usk_a
		pubkey = self.pubkey
		noClients = len(usk)

		# Generate labels and secrets for the matrix coefficients with usk_setup
		lastb = 0
		bA = np.arange(lastb,lastb+n*n).reshape(n,n) # use dtype = object if there are issues with int64
		lastb = lastb+n*n
		self.bA = util_fpv.voff_gen(pubkey,bA,usk[0])
		self.enc_bA = util_fpv.vencrypt(self.mpk,self.bA)
		bB = np.arange(lastb,lastb+n*m).reshape(n,m)
		lastb = lastb+n*m
		self.bB = util_fpv.voff_gen(pubkey,bB,usk[0])
		self.enc_bB = util_fpv.vencrypt(self.mpk,self.bB)
		bC = np.arange(lastb,lastb+p*n).reshape(p,n)
		lastb = lastb+p*n
		self.bC = util_fpv.voff_gen(pubkey,bC,usk[0])
		self.enc_bC = util_fpv.vencrypt(self.mpk,self.bC)
		bK = np.arange(lastb,lastb+m*n).reshape(m,n)
		lastb = lastb+m*n
		self.bK = util_fpv.voff_gen(pubkey,bK,usk[0])
		self.enc_bK = util_fpv.vencrypt(self.mpk,self.bK)
		bL = np.arange(lastb,lastb+n*p).reshape(n,p)
		lastb = lastb+n*p
		self.bL = util_fpv.voff_gen(pubkey,bL,usk[0])
		self.enc_bL = util_fpv.vencrypt(self.mpk,self.bL)

		# Generate labels and secrets for the measurements (and initial state) for the clients
		lastb = 0 # Changed the key to the sensor client's key
		self.bz = np.zeros((p,T), dtype=object)
		for k in range(T):
			bz = np.arange(lastb,lastb+p)
			lastb = lastb + p
			self.bz[:,k] = util_fpv.voff_gen(pubkey,bz,usk[1]) 
		self.enc_bz = util_fpv.vencrypt(self.mpk,self.bz)
		
		# Generate labels and secrets for the reference values
		# lastb = 0 
		bxr = np.arange(lastb,lastb+n, dtype=object)
		self.bxr= util_fpv.voff_gen(pubkey,bxr,usk[1]) 	
		self.enc_bxr = util_fpv.vencrypt(self.mpk,self.bxr)	

		# lastb = 0 
		bur = np.arange(lastb,lastb+m, dtype=object)
		self.bur= util_fpv.voff_gen(pubkey,bur,usk[1]) 	
		self.enc_bur = util_fpv.vencrypt(self.mpk,self.bur)	
		
		# lastb = 0 
		self.bx = np.zeros((n,T), dtype=object)
		for k in range(T):
			bx = np.arange(lastb,lastb+n)
			lastb = lastb + n
			self.bx[:,k] = util_fpv.voff_gen(pubkey,bx,usk[1]) 	
		self.enc_bx = util_fpv.vencrypt(self.mpk,self.bx)	

		# Generate labels and secrets for the refreshed values with usk_a if flag = 1 or with usk_setup if flag = 0
		if (flag == 1):
			lastb = 0 # Changed the key to the actuator's key
			# Gamma = A + BK - LCA - LCBK
			bGamma = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1	
			self.bGamma = util_fpv.voff_gen(pubkey,bGamma,usk_a)
			self.enc_bGamma = util_fpv.vencrypt(self.mpk,self.bGamma)
			# Gamma2 = BK - LCBK
			bGamma2 = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1			
			self.bGamma2 = util_fpv.voff_gen(pubkey,bGamma2,usk_a)
			self.enc_bGamma2 = util_fpv.vencrypt(self.mpk,self.bGamma2)
			# Gamma3 = B - LCB
			bGamma3 = np.arange(lastb+1,lastb+n*m+1).reshape(n,m)
			lastb = lastb+n*m+1			
			self.bGamma3 = util_fpv.voff_gen(pubkey,bGamma3,usk_a)
			self.enc_bGamma3 = util_fpv.vencrypt(self.mpk,self.bGamma3)		
		else:
			lastb = 0 # Changed the key to the actuator's key
			# Gamma = A + BK - LCA - LCBK
			bGamma = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1	
			self.bGamma = util_fpv.voff_gen(pubkey,bGamma,usk[0])
			self.enc_bGamma = util_fpv.vencrypt(self.mpk,self.bGamma)
			# Gamma2 = BK - LCBK
			bGamma2 = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1			
			self.bGamma2 = util_fpv.voff_gen(pubkey,bGamma2,usk[0])
			self.enc_bGamma2 = util_fpv.vencrypt(self.mpk,self.bGamma2)
			# Gamma3 = B - LCB
			bGamma3 = np.arange(lastb+1,lastb+n*m+1).reshape(n,m)
			lastb = lastb+n*m+1			
			self.bGamma3 = util_fpv.voff_gen(pubkey,bGamma3,usk[0])
			self.enc_bGamma3 = util_fpv.vencrypt(self.mpk,self.bGamma3)		

		## The commented parts are needed when we want to refresh Gref and uref
		self.bGref = np.dot(self.bGamma3,self.bur) - np.dot(self.bGamma2,self.bxr)
		self.buref = - np.dot(self.bK,self.bxr)	

		bxkr = np.zeros((n,T), dtype=object)
		bKx = np.zeros((m,T), dtype=object)
		for k in range(0,T):
			if k > 0:
				bxkr[:,k-1] = np.dot(self.bGamma,self.bx[:,k-1])+np.dot(self.bL,self.bz[:,k-1]) + self.bGref
			bKx[:,k] = np.dot(self.bK,self.bx[:,k]) + self.buref
		self.bxkr = bxkr
		self.bKx = bKx

	def genProgramSecrets(self):
		# Generate result secret for Gamma
		n = self.n; m = self.m; p = self.p
		# fbGamma = np.zeros((m,n), dtype=object)
		fbGamma = np.dot(np.eye(n,dtype = object) - np.dot(self.bL,self.bC),self.bA + np.dot(self.bB,self.bK))
		self.fbGamma = fbGamma
		self.enc_fbGamma = util_fpv.vencrypt(self.mpk,self.fbGamma)
		# Gamma = A + BK - LCA - LCBK

		fbGamma2 = np.dot(np.eye(n,dtype = object) - np.dot(self.bL,self.bC),np.dot(self.bB,self.bK))
		self.fbGamma2 = fbGamma2
		self.enc_fbGamma2 = util_fpv.vencrypt(self.mpk,self.fbGamma2)
		# Gamma = BK - LCBK

		fbGamma3 = np.dot(np.eye(n,dtype = object) - np.dot(self.bL,self.bC),self.bB,)
		self.fbGamma3 = fbGamma3
		self.enc_fbGamma3 = util_fpv.vencrypt(self.mpk,self.fbGamma3)
		# Gamma = B - LCB

		# Generate intermediate secrets for Gamma: BK, LCA, LCBK
		bLC = np.zeros((n,n), dtype=object); bLA = np.zeros((n,p,n,n), dtype=object); 
		bCA = np.zeros((p,n), dtype=object); bLCA = np.zeros((n,n), dtype=object);
		bLB = np.zeros((n,p,n,m), dtype=object); bCB = np.zeros((p,m), dtype=object); 
		bBK = np.zeros((n,n), dtype=object); bLCB = np.zeros((n,m), dtype=object); 
		bGamma3K = np.zeros((n,n), dtype=object)

		for i in range(n):
			for l in range(n):
				bBK[i,l] = sum(self.bB[i][j]*self.bK[j][l] for j in range(m))
		self.bBK = bBK # the final product should be kept in plaintext		
		for i in range(n):
			for l in range(n):
				bLC[i,l] = sum(self.bL[i][j]*self.bC[j][l] for j in range(p))
		# print('bLC ',bLC)
		self.enc_bLC = util_fpv.vencrypt(self.mpk,bLC)
		for i in range(n):
			for j in range(p):
				for l in range(n):
					for t in range(n):
						bLA[i,j,l,t] = self.bL[i][j]*self.bA[l][t]
		self.enc_bLA = util_fpv.encrypt_multi_dim_np(self.mpk,bLA)
		# print('bLA ', bLA)
		for i in range(p):
			for l in range(n):
				bCA[i,l] = sum(self.bC[i][j]*self.bA[j][l] for j in range(n))
		self.enc_bCA = util_fpv.vencrypt(self.mpk,bCA)
		# print('bCA ',bCA)
		for i in range(n):
			for l in range(n):
				bLCA[i,l] = sum(bLC[i,j]*self.bA[j][l] for j in range(n))
		self.bLCA = bLCA # the final product should be kept in plaintext
		for i in range(n):
			for j in range(p):
				for l in range(n):
					for t in range(m):
						bLB[i,j,l,t] = self.bL[i][j]*self.bB[l][t]
		self.enc_bLB = util_fpv.encrypt_multi_dim_np(self.mpk,bLB)
		for i in range(p):
			for l in range(m):
				bCB[i,l] = sum(self.bC[i][j]*self.bB[j][l] for j in range(n))
		self.enc_bCB = util_fpv.vencrypt(self.mpk,bCB)
		self.enc_bBK = util_fpv.vencrypt(self.mpk,bBK)
		for i in range(n):
			for l in range(m):
				bLCB[i,l] = sum(bLC[i,j]*self.bB[j][l] for j in range(n))
		self.bLCB = bLCB # the final product should be kept in plaintext
		self.enc_bLCB = util_fpv.vencrypt(self.mpk,bLCB)

		for i in range(n):
			for l in range(n):
				bGamma3K[i,l] = sum(self.bGamma3[i][j]*self.bK[j][l] for j in range(m))
		self.bGamma3K = bGamma3K # the final product should be kept in plaintext	


	def refresh_Gamma3(self,blinded_Gamma3):
		lf = DEFAULT_PRECISION

		old_bGamma3 =  self.bLCB*(-1) 
		Gammar3 = util_fpv.von_dec(self.privkey,blinded_Gamma3)+old_bGamma3
		new_Gamma3 = util_fpv.von_enc(self.pubkey,util_fpv.vfp(Gammar3,-2*lf),self.bGamma3)

		return 	new_Gamma3


	def refresh_Gamma1_2(self,blinded_Gamma,blinded_Gamma2):
		lf = DEFAULT_PRECISION
		old_bGamma = (np.dot(self.bGamma3K,2**lf*np.eye(self.n,dtype = object)) - self.bLCA)
		Gammar = util_fpv.von_dec(self.privkey,blinded_Gamma)+old_bGamma
		new_Gamma = util_fpv.von_enc(self.pubkey,util_fpv.vfp(Gammar,-2*lf),self.bGamma)

		Gammar2 = util_fpv.von_dec(self.privkey,blinded_Gamma2,self.bGamma3K)
		new_Gamma2 = util_fpv.von_enc(self.pubkey,util_fpv.vfp(Gammar2,-lf),self.bGamma2)

		return 	new_Gamma, new_Gamma2

	def refresh_constants(self,blinded_Gref,blinded_uref):
		lf = DEFAULT_PRECISION
		old_bGref = np.dot(self.bGamma3,self.bur) - np.dot(self.bGamma2,self.bxr)
		Gref = util_fpv.von_dec(self.privkey,blinded_Gref,old_bGref)
		new_Gref = util_fpv.von_enc(self.pubkey,Gref,self.bGref)

		old_buref = - np.dot(self.bK,self.bxr)
		uref = util_fpv.von_dec(self.privkey,blinded_uref) + old_buref
		new_uref = util_fpv.von_enc(self.pubkey,uref,self.buref)	

		return new_Gref, new_uref

	def refresh_xk(self,blinded_xk,k):
		bxkr = self.bxkr[:,k-1]
		# xkr = util_fpv.von_dec(self.privkey,blinded_xk)+bxkr
		# print('act x%d: '%(k+1), util_fpv.vfp(xkr,-2*DEFAULT_PRECISION)'')
		## if Gref is not a full LabHE encryption, but rather [[Gref - bGref]], then perform
		# bxk = bxkr + self.bGref
		xkr = util_fpv.von_dec(self.privkey,blinded_xk,bxkr)
		return util_fpv.von_enc(self.pubkey,util_fpv.vfp(xkr,-DEFAULT_PRECISION),self.bx[:,k])		

	def getInput(self,enc_uk,k):
		lf = DEFAULT_PRECISION
		bKx = self.bKx[:,k]
		uk = util_fpv.vretrieve_fp(util_fpv.von_dec(self.privkey,enc_uk)+bKx,2*lf)
		## if uref is not a full LabHE encryption, but rather [[uref - buref]], then perform
		# bKx = bKx + buref
		# uk = util_fpv.vretrieve_fp(util_fpv.von_dec(self.privkey,enc_uk,bKx),2*lf)

		return uk

def mult3Matrix(A,B,C,bAB,bAC,bBC):
	"""More memory-efficient than using the scalar product mlt3, which would require storing all the combinations 
	of indices in all matrix products, rather than summing over them.
	A,B,C are LabHE encryptions and AB,BC are matrices of the Paillier encryptions of the products of the secrets"""
	n1 = len(A)
	n2 = len(B)
	n3 = len(C)
	n4 = len(C[0])
	pA = [[x.ciphertext[0] for x in y] for y in A]; pA = np.array(pA, dtype=object)
	pB = [[x.ciphertext[0] for x in y] for y in B]; pB = np.array(pB, dtype=object)
	pC = [[x.ciphertext[0] for x in y] for y in C]; pC = np.array(pC, dtype=object)
	pProdAB = np.zeros((n1,n3), dtype=object); pProdBC = np.zeros((n2,n4), dtype=object)
	pProdAB = np.dot(pA,pB,pProdAB); pProdBC = np.dot(pB,pC,pProdBC)
	pProdABC = np.zeros((n1,n4), dtype=object)
	pProdABC = np.dot(pProdAB,pC,pProdABC)
	bA = [[x.ciphertext[1] for x in y] for y in A]; 
	bB = [[x.ciphertext[1] for x in y] for y in B]; 
	bC = [[x.ciphertext[1] for x in y] for y in C]; 
	enc_elem = np.ndarray((n1,n4), dtype=paillier.EncryptedNumber)
	for i in range(n1):
		for l in range(n4):
			# # Separate sums, for readibility
			# enc_elem[i,l] = (pProdABC[i][l] + sum(pA[i][j]*bBC[j][l] for j in range(n2)) +
			# 	sum(bAC[i][j][k][l]*pB[j][k] for j in range(n2) for k in range(n3)) +
			# 	sum(bAB[i][k]*pC[k][l] for k in range(n3))+ sum(int(pProdAB[i,k])*bC[k][l] for k in range(n3)) +
			# 	sum(pA[i][j]*bB[j][k]*pC[k][l] for j in range(n2) for k in range(n3)) +
			# 	sum(bA[i][j]*int(pProdBC[j,l]) for j in range(n2)))
			
			# Different elements gathered in the coresponding sums, for efficiency
			enc_elem[i,l] = (pProdABC[i][l] + 
				sum(pA[i][j]*bBC[j][l] + bA[i][j]*pProdBC[j,l] for j in range(n2)) +
				sum(bAC[i][j][k][l]*pB[j][k] + pA[i][j]*bB[j][k]*pC[k][l] for j in range(n2) for k in range(n3)) +
				sum(bAB[i][k]*pC[k][l] + pProdAB[i,k]*bC[k][l] for k in range(n3)))
	return enc_elem

def mult4Matrix(A,B,C,D,bABC,bABD,bACD,bBCD,bAB,bAC,bAD,bBC,bBD,bCD):
	"""More memory-efficient than using the scalar product mlt4, which would require storing all the combinations 
	of indices in all matrix products, rather than summing over them.
	A,B,C,D are LabHE encryptions and AB,BC are matrices of the Paillier encryptions of the products of the secrets"""
	n1 = len(A)
	n2 = len(B)
	n3 = len(C)
	n4 = len(D)
	n5 = len(D[0])
	pA = [[x.ciphertext[0] for x in y] for y in A]; pA = np.array(pA, dtype=object)
	pB = [[x.ciphertext[0] for x in y] for y in B]; pB = np.array(pB, dtype=object)
	pC = [[x.ciphertext[0] for x in y] for y in C]; pC = np.array(pC, dtype=object)
	pD = [[x.ciphertext[0] for x in y] for y in D]; pD = np.array(pD, dtype=object)	
	pProdAB = np.zeros((n1,n3), dtype=object); pProdBC = np.zeros((n2,n4), dtype=object)
	pProdCD = np.zeros((n3,n5), dtype=object); 
	pProdAB = np.dot(pA,pB,pProdAB); pProdBC = np.dot(pB,pC,pProdBC)
	pProdCD = np.dot(pC,pD,pProdCD)
	pProdABC = np.zeros((n1,n4), dtype=object); pProdBCD = np.zeros((n2,n5), dtype=object)
	pProdABC = np.dot(pProdAB,pC,pProdABC); pProdBCD = np.dot(pProdBC,pD,pProdBCD)
	pProdABCD = np.zeros((n1,n5), dtype=object)
	pProdABCD = np.dot(pProdABC,pD,pProdABCD);
	bA = [[x.ciphertext[1] for x in y] for y in A]; 
	bB = [[x.ciphertext[1] for x in y] for y in B]; 
	bC = [[x.ciphertext[1] for x in y] for y in C]; 
	bD = [[x.ciphertext[1] for x in y] for y in D]; 
	enc_elem = np.ndarray((n1,n5), dtype=paillier.EncryptedNumber)
	for i in range(n1):
		for t in range(n5):
			# # Separate sums, for readibility
			# enc_elem[i,t] = (pProdABCD[i][t] + sum(pProdABC[i][l]*bD[l][t] for l in range(n4)) +
			# 	sum(bA[i][j]*pProdBCD[j][t] for j in range(n2)) + 
			# 	sum(pProdAB[i][k]*bC[k][l]*pD[l][t] for k in range(n3) for l in range(n4)) +
			# 	sum(pA[i][j]*bB[j][k]*pProdCD[k][t] for j in range(n2) for k in range(n3)) +
			# 	sum(pProdAB[i][k]*bCD[k][t] for k in range(n3)) +
			# 	sum(pA[i][j]*bBD[j][k][l][t]*pC[k][l] for j in range(n2) for k in range(n3) for l in range(n4)) +
			# 	sum(pA[i][j]*bBC[j][l]*pD[l][t] for j in range(n2) for l in range(n4)) + 
			# 	sum(bAD[i][j][l][t]*pProdBC[j][l] for j in range(n2) for l in range(n4)) +
			# 	sum(bAC[i][j][k][l]*pB[j][k]*pD[l][t] for j in range(n2) for k in range(n3) for l in range(n4)) +
			# 	sum(bAB[i][k]*pProdCD[k][t] for k in range(n3)) + 
			# 	sum(bABC[i][l]*pD[l][t] for l in range(n4)) + 
			# 	sum(pA[i][j]*bBCD[j][t] for j in range(n2)) +
			# 	sum(bABD[i][k][l][t]*pC[k][l] for k in range(n3) for l in range(n4)) + 
			# 	sum(bACD[i][j][k][t]*pB[j][k] for j in range(n2) for k in range(n3)))

			# Different elements gathered in the coresponding sums, for efficiency
			enc_elem[i,t] = (pProdABCD[i][t] + 
				sum(bA[i][j]*pProdBCD[j][t] + pA[i][j]*bBCD[j][t] for j in range(n2)) + 
				sum(pProdAB[i][k]*bCD[k][t] + bAB[i][k]*pProdCD[k][t] for k in range(n3)) +
				sum(pProdABC[i][l]*bD[l][t] + bABC[i][l]*pD[l][t] for l in range(n4)) +
				sum(bACD[i][j][k][t]*pB[j][k] + pA[i][j]*bB[j][k]*pProdCD[k][t]
					for j in range(n2) for k in range(n3)) +
				sum(pProdAB[i][k]*bC[k][l]*pD[l][t] + bABD[i][k][l][t]*pC[k][l] 
					for k in range(n3) for l in range(n4)) +
				sum(pA[i][j]*bBC[j][l]*pD[l][t] + bAD[i][j][l][t]*pProdBC[j][l] 
					for j in range(n2) for l in range(n4)) +
				sum(pA[i][j]*bBD[j][k][l][t]*pC[k][l] + bAC[i][j][k][l]*pB[j][k]*pD[l][t] 
					for j in range(n2) for k in range(n3) for l in range(n4))
				)
	return enc_elem

