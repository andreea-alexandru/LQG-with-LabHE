# LQG-with-LabHE

This repository contains Python sample code for the protocols described in the paper “Encrypted LQG using Labeled Homomorphic Encryption”. 

The main function is in runLQG.py where different parameters can be set. The architecture contains clients, which are sub-systems of a plant, that collect private measurements, a setup, which has the model of a plant, a cloud server, which performs the estimation and control action computation on the encrypted model and measurements, and an actuator, which has the master key and receives the control actions after helping the cloud with the encrypted computations. Data for a temperature control application and random instances can be found in the Data folder.

The exact description, architecture and more details can be found in the paper “Encrypted LQG using Labeled Homomorphic Encryption”. The labeled homomorphic encryption scheme is described in the paper “Labeled Homomorphic Encryption: Scalable and Privacy-Preserving Processing of Outsourced Data” by Manuel Barbosa, Dario Catalano, and Dario Fiore https://eprint.iacr.org/2017/326. The underlying additively homomorphic encryption scheme we use is the Paillier cryptosystem.
