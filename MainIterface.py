import rpyc
import sys
import IPython
import numpy as np
import rpyc
c = rpyc.connect("localhost", 18861,config={ 'allow_public_attrs': True,"allow_all_attrs":True, "allow_setattr":True,"allow_pickle":True})
a = c.root

a.setNPLIB(np)
Inonsym = np.matrix([[1, 0.002, -0.002],[0,0.4, -0.001],[0, 0, 0.3]])
#Inonsym = np.matrix([[3, 0.02, 0],[0,1, 0],[0, 0, 0.3]])


Imat = 0.5 * (Inonsym.transpose() + Inonsym)
T_s = 1/2000
Tend = 15
a.initProp_RigidBodyRot(Imat, T_s, Tend)

x0  = np.matrix([0,0,0]).transpose()
dx0  = np.matrix([1,-2,0]).transpose()
ddx0  = np.matrix([0,0,0]).transpose()
a.setInitialCond(x0,dx0,ddx0)
a.runFreeSim()
#IPython.embed()