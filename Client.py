#!/usr/bin/env python3
# utf-8

from gmpy2 import mpz
import paillier
import numpy as np
import time
import os
import util_fpv
import LabHE

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


class Client:
	def __init__(self, key_length=DEFAULT_SEEDSIZE):
		key = gmpy2.mpz_urandomb(gmpy2.random_state(),key_length)
		self.usk = key

	#### Make sure the products of secrets does not exceed N/3!
	def Plant(self,T,folder,A=None,B=None,C=None,F=None,K=None,L=None,x0=None,z1=None,W=None,V=None):
		self.T = T
		if A is None:
			# the vectors are read as rows regardless if they are columns or not
			A = np.loadtxt("Data/"+folder+"/A.txt", delimiter=',', dtype = float)
			B = np.loadtxt("Data/"+folder+"/B.txt", delimiter=',', dtype = float)
			C = np.loadtxt("Data/"+folder+"/C.txt", delimiter=',', dtype = float)
			F = np.loadtxt("Data/"+folder+"/F.txt", delimiter=',', dtype = float)
			K = np.loadtxt("Data/"+folder+"/K.txt", delimiter=',', dtype = float)
			L = np.loadtxt("Data/"+folder+"/L.txt", delimiter=',', dtype = float)
			W = np.loadtxt("Data/"+folder+"/W.txt", delimiter=',', dtype = float)		
			V = np.loadtxt("Data/"+folder+"/V.txt", delimiter=',', dtype = float)

		size = np.shape(A)
		if len(size) == 2:
			self.n = size[0]
			if len(size) == 0:
				self.n == 1
		size = np.shape(B)
		if self.n == 1:
			if len(size) == 0:
				self.m = 1
			else:
				self.m = size
		else:
			if len(size) == 2:
				self.m = size[1]
			else:
				if len(size) == 1:
					self.m = 1
		size = np.shape(C)
		if self.n == 1:
			if len(size) == 0:
				self.p = 1
			else:
				self.p = size
		else:
			if len(size) == 2:
				self.p = size[0]
			else:
				if len(size) == 1:
					self.p = 1
		A = A.reshape(self.n,self.n); B = B.reshape(self.n,self.m); C = C.reshape(self.p,self.n)
		K = K.reshape(self.m,self.n); L = L.reshape(self.n,self.p)
		self.d, d = np.shape(W)	
		self.A = A; self.B = B; self.C = C
		self.F = F; self.K = K; self.L = L
		self.W = W; self.V = V; 

		self.Af = util_fpv.vfp(A); self.Bf = util_fpv.vfp(B); self.Cf = util_fpv.vfp(C)
		self.Kf = util_fpv.vfp(K); self.Lf = util_fpv.vfp(L)
		self.Wf = util_fpv.vfp(W); self.Vf = util_fpv.vfp(V)

	def getPubkey(self,pubkey):
		self.pubkey = pubkey
		self.mpk = pubkey.Pai_key

	def setPoints(self,folder,n,m,p,d,T,A,B,C,W,V,F):
		self.n = n; self.m = m; self.p = p; self.T = T; self.d = d
		x0 = np.loadtxt("Data/"+folder+"/x0.txt", delimiter=',', dtype = float)
		xr = np.loadtxt("Data/"+folder+"/xr.txt", delimiter=',', dtype = float)
		ur = np.loadtxt("Data/"+folder+"/ur.txt", delimiter=',', dtype = float)

		x0 = x0.reshape(n)
		xr = xr.reshape(n)
		ur = ur.reshape(m)

		self.x = x0
		self.x_series = np.zeros((n,T+1))
		self.x_series[:,0] = x0
		self.xr = xr; self.ur = ur
		self.xf = util_fpv.vfp(x0); 
		self.xrf = util_fpv.vfp(xr); self.urf = util_fpv.vfp(ur);

		self.A = A; self.B = B; self.C = C
		self.W = W; self.V = V; self.F = F

	def genLabels(self):
		n = self.n; m = self.m; p = self.p; T = self.T
		usk = self.usk
		pubkey = self.pubkey
		mpk = self.mpk

		# Generate labels and secrets for the measurements (and initial state) for the clients
		lastb = 0 # Changed the key to the sensor client's key
		# The measurements start at time 1 but we use labels from 0
		self.bz = np.zeros((p,T), dtype=object)
		for k in range(T):
			bz = np.arange(lastb,lastb+p, dtype=object)
			lastb = lastb + p
			self.bz[:,k] = util_fpv.voff_gen(pubkey,bz,usk)
		self.enc_bz = util_fpv.vencrypt(mpk,self.bz)

		# Generate labels and secrets for the reference values
		# lastb = 0 
		bxr = np.arange(lastb,lastb+n, dtype=object)
		self.bxr = util_fpv.voff_gen(pubkey,bxr,usk)
		self.enc_bxr = util_fpv.vencrypt(mpk,self.bxr)	

		# lastb = 0 
		bur = np.arange(lastb,lastb+m, dtype=object)
		self.bur = util_fpv.voff_gen(pubkey,bur,usk)
		self.enc_bur = util_fpv.vencrypt(mpk,self.bur)	

		# Generate labels and secrets for the initial state
		# lastb = 0 
		bx = np.arange(lastb,lastb+n, dtype=object)
		self.bx = util_fpv.voff_gen(pubkey,bx,usk)
		self.enc_bx = util_fpv.vencrypt(mpk,self.bx)

	def genLabelsSetup(self, flag):
		n = self.n; m = self.m; p = self.p; T = self.T
		usk = self.usk
		pubkey = self.pubkey
		mpk = self.mpk
		# Generate labels and secrets for the matrix coefficients
		lastb = 0
		bA = np.arange(lastb,lastb+n*n).reshape(n,n) # use dtype = object if there are issues with int64
		lastb = lastb+n*n
		self.bA = util_fpv.voff_gen(pubkey,bA,usk)
		self.enc_bA = util_fpv.vencrypt(self.mpk,self.bA)

		bB = np.arange(lastb,lastb+n*m).reshape(n,m)
		lastb = lastb+n*m
		self.bB = util_fpv.voff_gen(pubkey,bB,usk)
		self.enc_bB = util_fpv.vencrypt(self.mpk,self.bB)

		bC = np.arange(lastb,lastb+p*n).reshape(p,n)
		lastb = lastb+p*n
		self.bC = util_fpv.voff_gen(pubkey,bC,usk)
		self.enc_bC = util_fpv.vencrypt(self.mpk,self.bC)

		bK = np.arange(lastb,lastb+m*n).reshape(m,n)
		lastb = lastb+m*n
		self.bK = util_fpv.voff_gen(pubkey,bK,usk)
		self.enc_bK = util_fpv.vencrypt(self.mpk,self.bK)

		bL = np.arange(lastb,lastb+n*p).reshape(n,p)
		lastb = lastb+n*p
		self.bL = util_fpv.voff_gen(pubkey,bL,usk)
		self.enc_bL = util_fpv.vencrypt(self.mpk,self.bL)

		if (flag == 0):
			lastb = 0 # Changed the key to the actuator's key
			# Gamma = A + BK - LCA - LCBK
			bGamma = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1	
			self.bGamma = util_fpv.voff_gen(pubkey,bGamma,usk)
			self.enc_bGamma = util_fpv.vencrypt(self.mpk,self.bGamma)

			# Gamma2 = BK - LCBK
			bGamma2 = np.arange(lastb+1,lastb+n*n+1).reshape(n,n)
			lastb = lastb+n*n+1			
			self.bGamma2 = util_fpv.voff_gen(pubkey,bGamma2,usk)
			self.enc_bGamma2 = util_fpv.vencrypt(self.mpk,self.bGamma2)

			# Gamma3 = B - LCB
			bGamma3 = np.arange(lastb+1,lastb+n*m+1).reshape(n,m)
			lastb = lastb+n*m+1			
			self.bGamma3 = util_fpv.voff_gen(pubkey,bGamma3,usk)
			self.enc_bGamma3 = util_fpv.vencrypt(self.mpk,self.bGamma3)	

			self.Gamma = np.dot(np.eye(n, dtype = object) - np.dot(self.L,self.C),self.A + np.dot(self.B,self.K))
			self.Gamma2 = np.dot(np.eye(n, dtype = object) - np.dot(self.L,self.C),np.dot(self.B,self.K))
			self.Gamma3 = np.dot(np.eye(n, dtype = object) - np.dot(self.L,self.C),self.B)


	def encryptMatrices(self,flag):
		pubkey = self.pubkey
		self.enc_A = util_fpv.von_enc(pubkey,self.Af,self.bA,self.enc_bA)
		# print('A: ',util_fpv.retrieve_fp_matrix(util_fpv.on_dec_mat(privkey,enc_A,act.bA)))	
		self.enc_C = util_fpv.von_enc(pubkey,self.Cf,self.bC,self.enc_bC)
		# print('C: ',util_fpv.retrieve_fp_matrix(util_fpv.on_dec_mat(privkey,enc_C,act.bC)))
		self.enc_L = util_fpv.von_enc(pubkey,self.Lf,self.bL,self.enc_bL)
		# print('L: ',util_fpv.retrieve_fp_matrix(util_fpv.on_dec_mat(privkey,enc_L,act.bL)))
		self.enc_B = util_fpv.von_enc(pubkey,self.Bf,self.bB,self.enc_bB)
		# print('B: ',util_fpv.retrieve_fp_matrix(util_fpv.on_dec_mat(privkey,enc_B,act.bB)))
		self.enc_K = util_fpv.von_enc(pubkey,self.Kf,self.bK,self.enc_bK)

		if flag == 0:
			self.enc_Gamma = util_fpv.von_enc(pubkey,util_fpv.vfp(self.Gamma),self.bGamma,self.enc_bGamma)
			self.enc_Gamma2 = util_fpv.von_enc(pubkey,util_fpv.vfp(self.Gamma2),self.bGamma2,self.enc_bGamma2)
			self.enc_Gamma3 = util_fpv.von_enc(pubkey,util_fpv.vfp(self.Gamma3),self.bGamma3,self.enc_bGamma3)

	def encryptSetPoints(self):
		pubkey = self.pubkey
		self.enc_xr = util_fpv.von_enc(pubkey,self.xrf,self.bxr,self.enc_bxr)
		self.enc_ur = util_fpv.von_enc(pubkey,self.urf,self.bur,self.enc_bur)
		self.enc_x0 = util_fpv.von_enc(pubkey,self.xf,self.bx,self.enc_bx)

	def getMeasurement(self,k):
		z = np.add(np.dot(self.C,self.x),np.random.multivariate_normal([0]*self.p, self.V, tol=1e-6))
		# z = np.dot(self.C,self.x)	# no noise
		# z = np.array(z, dtype=object)
		self.z = z
		# print("Measurement z%d: "%(k+1),["%.5f"% i for i in z])
		enc_bz = [item[k] for item in self.enc_bz]
		enc_z = util_fpv.von_enc(self.pubkey,util_fpv.vfp(z),self.bz[:,k],enc_bz)
		return enc_z


	def closedLoop(self,u,k):
		# print("Last input: ", ["%.5f"% i for i in u])
		# with np.errstate(invalid='ignore'): 
		# np.warnings.filterwarnings('ignore')
		x = np.add(np.dot(self.A,self.x),np.dot(self.B,u),np.dot(self.F,np.random.multivariate_normal([0]*self.d, self.W, tol=1e-6)))
		# x = np.dot(self.A,self.x) + np.dot(self.B,u) # no noise
		self.x = x
		self.x_series[:,k] = x
		self.xf = util_fpv.vfp(x)
		# print("Next state: ", ["%.5f"% i for i in self.x])


def main():
	client = Client()



# main()
if __name__ == '__main__':
	main()
