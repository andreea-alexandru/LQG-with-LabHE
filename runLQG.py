#!/usr/bin/env python3

import Cloud
import Client
import Actuator
import numpy as np
import util_fpv
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import os

# When changing default sizes, remember to change in all other classes: Client, Cloud, Actuator, util_fpv and keys in paillier and LabHE
DEFAULT_KEYSIZE = 1024
DEFAULT_SEEDSIZE = 32
DEFAULT_MSGSIZE = 48 
DEFAULT_PRECISION = 24
DEFAULT_SECURITYSIZE = 100
NETWORK_DELAY = 0 #0.01 # 10 ms

"""We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
The range N/3 < x < 2N/3 allows for overflow detection. The product of the hashes has to fit in the 
message space, products of secrets do not exceed N/3!"""


def memory_usage_resource():
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def main():
	""" Run instances of encrypted LQG with Labeled Homomorphic Encryption."""

	flag = 0 # The Setup computes gives the gain matrices Gamma to the cloud if flag = 0, otherwise, the cloud and actuator compute their encryption
	verbose = 0 # Print values of variables if verbose = 1, does nothing otherwise

	# folder = 'building_1_room' 
	# folder = 'Random_instance_5_1'
	folder = 'building_2_rooms'
	# folder = 'Random_instance_25_5'
	# folder = 'Random_instance_50_10'
	# folder = 'Random_instance_75_15'
	# folder = 'Random_instance_100_20'
	# folder = 'Random_instance_125_25'

	lf = DEFAULT_PRECISION
	T = 100
	setup = Client.Client()
	usk_setup = setup.usk
	setup.Plant(T,folder)
	n = setup.n; m = setup.m; p = setup.p; d = setup.d
	A = setup.Af; B = setup.Bf; C = setup.Cf
	K = setup.Kf; L = setup.Lf
	W = setup.Wf; V = setup.Vf; F = setup.F

	client = Client.Client()
	usk_client = client.usk
	client.setPoints(folder,n,m,p,d,T,setup.A,setup.B,setup.C,setup.W,setup.V,setup.F)
	x0 = client.xf;	
	xr = client.xr; ur = client.ur
	xrf = client.xrf; urf = client.urf
	plaintext_x_series = np.zeros((n,T))

	act = Actuator.Actuator(usk_setup,usk_client)
	privkey = act.privkey
	msk = privkey.msk; upk = privkey.upk
	pubkey = act.pubkey

	client.getPubkey(pubkey)
	setup.getPubkey(pubkey)

	start = time.time()
	setup.genLabelsSetup(flag)
	time_off_setup = time.time() - start
	print('Offline time for setup: ', "%.5f" % (time_off_setup), ' sec')
	start = time.time()

	client.genLabels()
	time_off_clients = time.time() - start
	print('Offline time for clients: ', "%.5f" % (time_off_clients), ' sec')
	start = time.time()

	cloud = Cloud.Cloud(act.pubkey,n,m,p,T)
	cloud.generateRandomness()
	time_off_cloud = time.time() - start
	print('Offline time for cloud: ', "%.5f" % (time_off_cloud), ' sec')
	start = time.time()

	act.genLabels(n,m,p,T,flag)
	
	if flag == 1:
		act.genProgramSecrets()
	time_off_act = time.time() - start
	print('Offline time for actuator: ', "%.5f" % (time_off_act), ' sec')
	start = time.time()

	setup.encryptMatrices(flag)
	time_in_setup = time.time() - start
	print('Online time to compute constants for setup: ', "%.5f" % (time_in_setup), ' sec')

	start = time.time()
	client.encryptSetPoints()
	time_in_clients = time.time() - start
	print('Online time to compute constants for clients: ', "%.5f" % (time_in_clients), ' sec')

	if flag == 1:
		start = time.time()
		cloud.getCoeff(setup.enc_A,setup.enc_B,setup.enc_C,setup.enc_L,setup.enc_K,client.enc_xr,client.enc_ur)
		cloud.getProgramSecrets(act.enc_bLC,act.enc_bLA,act.enc_bCA,act.enc_bLB,act.enc_bCB)

		blinded_enc_Gamma3 = cloud.computeGamma3()
		# The actuator needs to refresh Gamma3
		start_act = time.time()			
		blinded_enc_Gamma3 = act.refresh_Gamma3(blinded_enc_Gamma3)
		time_in_act = time.time() - start_act
		cloud.getGamma3(blinded_enc_Gamma3)
		blinded_enc_Gamma, blinded_enc_Gamma2 = cloud.computeProducts()
		# The actuator needs to refresh Gamma, Gamma2, Gamma3
		start_act = time.time()	
		blinded_enc_Gamma, blinded_enc_Gamma2 = act.refresh_Gamma1_2(blinded_enc_Gamma, blinded_enc_Gamma2)
		time_in_act = time.time() - start_act + time_in_act
		print('Online time to compute constants for actuator: ', "%.5f" % (time_in_act), ' sec')
		cloud.getGamma1_2(blinded_enc_Gamma, blinded_enc_Gamma2)
		cloud.computeConstants()
		time_in_cloud = time.time() - start - time_in_act
		print('Online time to compute constants for cloud: ', "%.5f" % (time_in_cloud), ' sec')

	if flag == 0:
		Gamma = setup.Gamma; Gamma2 = setup.Gamma2; Gamma3 = setup.Gamma3
		enc_Gamma = setup.enc_Gamma; enc_Gamma2 = setup.enc_Gamma2; enc_Gamma3 = setup.enc_Gamma3
	else:
		Gamma = np.dot(np.eye(n, dtype = object) - np.dot(setup.L,setup.C),setup.A + np.dot(setup.B,setup.K))
		Gamma2 = np.dot(np.eye(n, dtype = object) - np.dot(setup.L,setup.C),np.dot(setup.B,setup.K))
		Gamma3 = np.dot(np.eye(n, dtype = object) - np.dot(setup.L,setup.C),setup.B)
		enc_Gamma = util_fpv.von_enc(pubkey,util_fpv.vfp(Gamma),act.bGamma,act.enc_bGamma)
		enc_Gamma2 = util_fpv.von_enc(pubkey,util_fpv.vfp(Gamma2),act.bGamma2,act.enc_bGamma2)
		enc_Gamma3 = util_fpv.von_enc(pubkey,util_fpv.vfp(Gamma3),act.bGamma3,act.enc_bGamma3)

	Gref = np.dot(Gamma3, client.ur) - np.dot(Gamma2, client.xr)
	Gref_2 = 2**lf*Gref
	uref = ur - np.dot(setup.K, client.xr)
	uref_2 = np.dot(uref,2**lf*np.eye(m,dtype = object))

	cloud.enc_Gamma = enc_Gamma
	cloud.enc_Gamma2 = enc_Gamma2
	cloud.enc_Gamma3 = enc_Gamma3

	if flag == 0:
		time_in_act = 0
		start = time.time()
		cloud.getCoeff(setup.enc_A,setup.enc_B,setup.enc_C,setup.enc_L,setup.enc_K,client.enc_xr,client.enc_ur)
		cloud.computeConstants()
		time_in_cloud = time.time() - start
		print('Online time to compute constants for cloud: ', "%.5f" % (time_in_cloud), ' sec')

	if verbose == 1:
		hat_x = np.zeros((n,T+1))
		enc_x0 = util_fpv.von_enc(pubkey,x0,act.bx[:,0],util_fpv.vencrypt(pubkey.Pai_key,act.bx[:,0]))
		print('x0: ',util_fpv.vretrieve_fp(util_fpv.von_dec(privkey,enc_x0,
				util_fpv.vencrypt(pubkey.Pai_key,act.bx[:,0]))))

	start = time.time()

	k = 0
	cloud.xk = client.enc_x0
	enc_u0 = cloud.computeInput()
	u0 = act.getInput(enc_u0,k)
	if verbose == 1:
		plaintext_xk = client.x
		plaintext_x_series[:,k] = plaintext_xk
		hat_x[:,k] = util_fpv.vretrieve_fp(util_fpv.von_dec(privkey,cloud.xk,act.bx[:,k]))
		print('decrypted x%d: '%k, ["%.5f"% i for i in hat_x[:,k]])
		print('decrypted u%d: '%k, ["%.5f"% i for i in u0])
		print('u0: ', ["%.5f"% i for i in np.dot(setup.K,client.x) + uref])

	# Get next state and measurement
	client.closedLoop(u0,k+1)
	enc_zk = client.getMeasurement(k) # The measurement time is k+1 but the labels are generated for k
	if verbose == 1:
		plaintext_zk = client.z
		print('z1: ', ["%.5f"% i for i in plaintext_zk])
		print('decrypted z%d: '%(k+1), ["%.5f"% i for i in util_fpv.vretrieve_fp(util_fpv.von_dec(privkey,enc_zk))])

		print('Online time for iteration %d: '%k, "%.5f" % (time.time() - start), ' sec')

	time_act = np.zeros((T-1,1))
	time_cloud = np.zeros((T-1,1))
	time_client = np.zeros((T-1,1))

	# Start the control loop
	for k in range(1,T):
		start_it = time.time()
		blinded_xk = cloud.computeEstimate(cloud.xk,enc_zk)
		start_act = time.time()	
		enc_xk2 = act.refresh_xk(blinded_xk,k)
		end_act = time.time()-start_act
		cloud.getEstimate(enc_xk2)		# get xk

		if verbose == 1:
			hat_x[:,k] = util_fpv.vretrieve_fp(util_fpv.von_dec(privkey,cloud.xk,act.bx[:,k]))
			print('decrypted x%d: '%k, ["%.5f"% i for i in hat_x[:,k]])
			plaintext_xk = np.dot(Gamma,plaintext_xk)+np.dot(setup.L,plaintext_zk)+Gref
			plaintext_x_series[:,k] = plaintext_xk
			print('x%d: '%k, ["%.5f"% i for i in plaintext_xk])

		enc_uk = cloud.computeInput()

		start_act = time.time()
		uk = act.getInput(enc_uk,k)			# get uk
		time_act[k-1] = end_act + time.time() - start_act
		if verbose == 1:
			print('decrypted u%d: '%k, uk)
			print('u%d: '%k, ["%.5f"% i for i in np.dot(setup.K,plaintext_xk)+uref])

		client.closedLoop(uk,k+1)

		start_cl = time.time()
		enc_zk = client.getMeasurement(k)	# get zk+1
		time_client[k-1] = time.time() - start_cl
		if verbose == 1:
			plaintext_zk = client.z
			plaintext_zk = util_fpv.vretrieve_fp(util_fpv.von_dec(privkey,enc_zk,
					util_fpv.vencrypt(pubkey.Pai_key,act.bz[:,k])))
			print('z%d: '%(k+1), ["%.5f"% i for i in plaintext_zk])

		time_cloud[k-1] = time.time() - start_it - time_act[k-1] - time_client[k-1]

	print('Total online time for %d iterations: '%T, "%.5f" % (time.time() - start), ' sec')

	print('Mean online time for client for one iteration: ', " %.5f" % np.mean(time_client), ' sec')
	print('Mean online time for actuator for one iteration: ', " %.5f" % np.mean(time_act), ' sec')
	print('Mean online time for cloud for one iteration: ', " %.5f" % np.mean(time_cloud), ' sec')

	# print("%.5f, %.5f MB" % (memory_usage_resource(),memory_usage_psutil()))

	with open(os.path.abspath('results_'+str(DEFAULT_KEYSIZE)+'_'+str(DEFAULT_PRECISION)+'.txt'),'a+') as f: 
		f.write("%d, %d, %d, flag %d\n " % (n,m,T,flag));
		f.write("Offline time:\n setup: %.5f, client: %.5f, cloud: %.5f, actuator %.5f\n" % (time_off_setup, time_off_clients, time_off_cloud, time_off_act))
		f.write("Initialization time:\n setup: %.5f, client: %.5f, cloud: %.5f, actuator %.5f\n" % (time_in_setup, time_in_clients, time_in_cloud, time_in_act))
		f.write("Online time 1 iter:\n client: %.5f, cloud: %.5f, actuator %.5f\n" % (np.mean(time_client), np.mean(time_act), np.mean(time_cloud)))


	if folder == 'building_2_rooms' and verbose == 1:
		# evenly sampled time at 440s intervals
		Ts = 420
		t = Ts*np.arange(0., T)

		fig, ax = plt.subplots(figsize=(16,8))

		for i in range(int(n/2)):
			ax.plot(t, hat_x[i,:-1], 'C'+str(i), linewidth=2)
			ax.plot(t, hat_x[int(n/2)+i,:-1], 'C'+str(int(n/2)+i)+'--', linewidth=2)
			# ax.plot(t, plaintext_x_series[i,:],'C'+str(i)+'--',linewidth=2)

		ax.set(xlabel='Time (s)', ylabel='State x',
		       title='Evolution of the estimates')
		ax.grid()
		plt.legend(('$\hat x_1$', '$\hat x_6$', '$\hat x_2$', '$\hat x_7$', '$\hat x_3$', '$\hat x_8$',
			'$\hat x_4$', '$\hat x_9$', '$\hat x_5$', '$\hat x_{10}$'),
		           loc='lower right', shadow=True)
		fig.savefig("Figures/estimates.png")
		fig.show()

		fig2, ax = plt.subplots(figsize=(16,8))

		t = Ts*np.arange(0., T)
		for i in range(int(n/2)):
			ax.plot(t, client.x_series[i,:-1], 'C'+str(i), linewidth=2)
			ax.plot(t, client.x_series[int(n/2)+i,:-1], 'C'+str(int(n/2)+i)+'--', linewidth=2)
			# ax.plot(t, plaintext_x_series[i,:],'C'+str(i)+'--',linewidth=2)

		ax.set(xlabel='Time (s)', ylabel='State x',
		       title='Evolution of the true states')
		ax.grid()
		plt.legend(('$x_1$', '$x_6$', '$x_2$', '$x_7$', '$x_3$', '$x_8$',
			'$x_4$', '$x_9$', '$x_5$', '$x_{10}$'),
		           loc='lower right', shadow=True)
		fig.savefig("Figures/states.png")
		plt.show()

# main()
if __name__ == '__main__':
	main()
