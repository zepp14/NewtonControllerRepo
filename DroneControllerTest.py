import numpy as np
import matplotlib.pyplot as plt
import time
from PhysicsEngine import EulerMethod_Propogator, SystemParameter, hat

StateDim = 3
Ts = 1/3000
T_end = 10

def FirstDerivFunction(u_in , params, State, dState, ddState, ts=0):
    return dState

def SecondDerivFunction(u_in , params, State, dState, ddState, ts=0):
    g = params.g
    m = params.m
    dd_x = g*np.sin(u_in[1,0])
    dd_y = -g*np.sin(u_in[0,0])*np.cos(u_in[1,0])
    dd_z = u_in[2,0]/m - g*np.cos(u_in[0,0])*np.cos(u_in[1,0])
    return np.matrix([dd_x,dd_y,dd_z]).transpose()


def NewtonController(x, dx, ddx, params, u):
    T = params.T
    a = params.a
    JacobianFunc = params.J
    Jx = JacobianFunc(u, params.SysParams)
    print(Jx)

    u = -(a/T) * np.linalg.inv(Jx) @ (dx + T * ddx) + np.matrix([0,0,params.SysParams.g]).transpose()

    return u

def JacobianFunc(x,params):
    g = params.g
    m = params.m
    phi = x[0,0]
    Th = x[1,0]
    T = x[2,0]
    mat = np.matrix([[0.0, -g*np.cos(phi)*np.cos(Th), g*np.cos(Th)*np.sin(phi)],
                     [g*np.cos(Th), g*np.sin(phi)*np.sin(Th), g*np.sin(Th)*np.cos(Th)],
                     [0,0, 1/m]])

    return mat.transpose()

SysParams = SystemParameter()
SysParams.addProperty("g", 9.81)
SysParams.addProperty("m", 1.0)


ContParams = SystemParameter()
ContParams.addProperty("a",0.06)
ContParams.addProperty("T", 0.04)
ContParams.addProperty("J", JacobianFunc)
ContParams.addProperty("SysParams", SysParams)

PE = EulerMethod_Propogator(StateDim,Ts,T_end,FirstDerivFunction,SysParams,SecondDerivFunction,SysParams)

PE.InitialCondition_state  = np.matrix([0,0,0]).transpose()
PE.InitialCondition_d_state  = np.matrix([1,0,0]).transpose()
PE.InitialCondition_dd_state  = np.matrix([0,0,0]).transpose()

PE.ControllerFunction = NewtonController
PE.ControllerParam = ContParams
PE.Ts_controller = 1/100



u_in = np.matrix([[1],[0],[0]])
startT =  time.time()
PE.EulerLoop_w_Controller()
startEnd =  time.time()

print("Runtime: ", startEnd - startT )
fig, ax = plt.subplots(3,1)

ax[0].plot(PE.TimeVect, PE.StateVector[:,0])
ax[0].plot(PE.TimeVect, PE.StateVector[:,1])
ax[0].plot(PE.TimeVect, PE.StateVector[:,2])

ax[1].plot(PE.TimeVect, PE.dStateVector[:,0])
ax[1].plot(PE.TimeVect, PE.dStateVector[:,1])
ax[1].plot(PE.TimeVect, PE.dStateVector[:,2])

ax[2].plot(PE.TimeVect, PE.ddStateVector[:,0])
ax[2].plot(PE.TimeVect, PE.ddStateVector[:,1])
ax[2].plot(PE.TimeVect, PE.ddStateVector[:,2])




plt.show()

