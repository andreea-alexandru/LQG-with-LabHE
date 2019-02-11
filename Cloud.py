#!/usr/bin/env python3

from gmpy2 import mpz
import paillier
import numpy as np
import util_fpv
import LabHE
import Actuator
import Client

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

DEFAULT_KEYSIZE = 1024
DEFAULT_SEEDSIZE = 32
DEFAULT_MSGSIZE = 48 
DEFAULT_PRECISION = 24
DEFAULT_SECURITYSIZE = 100
NETWORK_DELAY = 0 #0.01 # 10 ms

"""We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
The range N/3 < x < 2N/3 allows for overflow detection. The product of the hashes has to fit in the 
message space, products of secrets do not exceed N/3!"""

class Cloud:
	def __init__(self,pubkey,n,m,p,T,l=DEFAULT_MSGSIZE,sigma=DEFAULT_SECURITYSIZE):
		self.pubkey = pubkey
		self.mpk = pubkey.Pai_key
		self.n = n; self.m = m; self.p = p; self.T = T
		self.l = l
		self.sigma = sigma

	def getCoeff(self,A,B,C,L,K,xr,ur):
		self.enc_A = A; self.enc_B = B; self.enc_C = C
		self.enc_L = L; self.enc_K = K
		self.xr = xr; self.ur = ur

	def getProgramSecrets(self,enc_bLC,enc_bLA,enc_bCA,enc_bLB,enc_bCB):
		# Get the extra secrets for the degree-3 and degree-4 multiplications
		self.enc_bLC = enc_bLC
		self.enc_bLA = enc_bLA
		self.enc_bCA = enc_bCA
		self.enc_bLB = enc_bLB
		self.enc_bCB = enc_bCB

	def generateRandomness(self):
		n = self.n; m = self.m; T = self.T
		state = gmpy2.random_state()
		rk = [gmpy2.mpz_urandomb(state,self.l+self.sigma) for i in range(n*T)]
		self.rk = rk
		self.rkf = util_fpv.vfp(rk,-DEFAULT_PRECISION)

	def computeGamma3(self):
		lf = DEFAULT_PRECISION
		# Compute LCB encrypted
		prod_LCB = Mult3Matrix(self.enc_L,self.enc_C,self.enc_B,self.enc_bLC,self.enc_bLB,self.enc_bCB)
		self.enc_LCB = prod_LCB

		# Compute Gamma3 = (I-LC)B = B - LCB, they have different precisions
		enc_Gamma3 = np.dot(self.enc_B,2**(2*lf)*np.eye(self.m,dtype = object)) - prod_LCB

		rGamma3 = [[gmpy2.mpz_urandomb(gmpy2.random_state(),self.l+2*lf+self.sigma) for i in range(self.m)] for j in range(self.n)]

		self.rGamma3 = util_fpv.vfp(rGamma3,-2*lf)

		return enc_Gamma3 + rGamma3

	def computeProducts(self):
		lf = DEFAULT_PRECISION
		# Compute LCA encrypted
		prod_LCA = Mult3Matrix(self.enc_L,self.enc_C,self.enc_A,self.enc_bLC,self.enc_bLA,self.enc_bCA)
		self.enc_LCA = prod_LCA

		# Compute Gamma2 = (I-LC)BK = BK - LCBK = Gamma3*K
		enc_Gamma2 = np.dot(self.enc_Gamma3,self.enc_K)

		rGamma2 = [[gmpy2.mpz_urandomb(gmpy2.random_state(),self.l+lf+self.sigma) for i in range(self.n)] for j in range(self.n)]

		self.rGamma2 = util_fpv.vfp(rGamma2,-lf)

		# Compute Gamma = (I-LC)(A+BK) = A + BK - LCA - LCBK, they have different precisions
		enc_Gamma = (np.dot(self.enc_A,2**(2*lf)*np.eye(self.n,dtype = object)) - prod_LCA + 
			np.dot(enc_Gamma2,2**lf*np.eye(self.n,dtype = object)))

		rGamma = [[gmpy2.mpz_urandomb(gmpy2.random_state(),self.l+2*lf+self.sigma) for i in range(self.n)] for j in range(self.n)]

		self.rGamma = util_fpv.vfp(rGamma,-2*lf)


		return enc_Gamma + rGamma, enc_Gamma2 + rGamma2

	def getGamma3(self,blinded_Gamma3):
		self.enc_Gamma3 = [[blinded_Gamma3[i][j] - self.rGamma3[i][j] for j in range(self.m)] for i in range(self.n)]

	def getGamma1_2(self,blinded_Gamma, blinded_Gamma2):
		self.enc_Gamma = [[blinded_Gamma[i][j] - self.rGamma[i][j] for j in range(self.n)] for i in range(self.n)]
		self.enc_Gamma2 = [[blinded_Gamma2[i][j] - self.rGamma2[i][j] for j in range(self.n)] for i in range(self.n)]
		
	def computeConstants(self):
		lf = DEFAULT_PRECISION
		enc_Gref = np.dot(self.enc_Gamma3, self.ur) - np.dot(self.enc_Gamma2, self.xr)				## 2**2lf

		enc_uref = np.dot(self.ur,2**lf*np.eye(self.m,dtype = object)) - np.dot(self.enc_K,self.xr) ## 2**2lf

		# return enc_Gref + rGref, enc_uref + ruref
		self.enc_Gref = enc_Gref
		self.enc_uref = enc_uref

	def getConstants(self,blinded_Gref, blinded_uref):
		self.enc_Gref = [blinded_Gref[i] - self.rGref[i] for i in range(self.n)]
		self.enc_uref = [blinded_uref[i] - self.ruref[i] for i in range(self.m)]


	def computeEstimate(self,xk_1,zk):
		n = self.n
		xk = np.dot(self.enc_Gamma,xk_1) + np.dot(self.enc_L,zk) + self.enc_Gref
		rk = self.rk[-n:]
		del self.rk[-n:]
		# return blinded estimate
		return xk + rk

	def getEstimate(self,blinded_xk):
		n = self.n
		rkf = self.rkf[-n:]
		self.rkf = self.rkf[:-n]		# Because it is passed through vfp, self.rkf is a ndarray, not a list like self.rk
		self.xk = [blinded_xk[i] - rkf[i] for i in range(self.n)]

	def computeInput(self):
		return self.enc_uref + np.dot(self.enc_K,self.xk)


def Mult3Matrix(A,B,C,bAB,bAC,bBC):
	"""More memory-efficient than using the scalar product mlt3, which would require storing all the combinations 
	of indices in all matrix products, rather than summing over them.
	A,B,C are LabHE encryptions and AB,BC are matrices of the Paillier encryptions of the products of the secrets"""
	lf = DEFAULT_PRECISION
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
