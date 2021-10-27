import numpy as np
import matplotlib.pyplot as plt
import time
from PhysicsEngine import EulerMethod_Propogator, SystemParameter, hat

StateDim = 3
Ts = 1/4000
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

def PID(x, dx, ddx, params, u):
    kp = params[0]
    kd = params[1]
    m = params[4]

    kpz = params[2]
    kdz = params[3]
    g = -9.81

    u = -(1/9.81)*((kp*x + kd*dx)  +  m* np.matrix([0,0,9.81]))

    u[1, 0] = -(1/g)*(kp * x[0,0] + kd*dx[0,0]) 
    u[0, 0] = (1/g) *(kp * x[1,0] + kd*dx[1,0]) 
    u[2, 0] = m*(g + kpz * x[2,0] + kdz*dx[2,0])
    

    mean = [0, 0, 0]
    cov =  0*np.eye(3)
    x = np.random.multivariate_normal(mean, cov)

    u[0,0] = np.clip(u[0,0], -np.pi/2.0, np.pi/2.0)

    
    u[1,0] = np.clip(u[1,0], -np.pi/2.0, np.pi/2.0)
    return -u + x


def NewtonController(x, dx, ddx, params, u):
    T = params.T
    a = params.a
    b = params.b
    m = params.SysParams.m
    JacobianFunc = params.J
    Jx = JacobianFunc(np.zeros((3,1)), params.SysParams)
    Jx = JacobianFunc(u, params.SysParams)
    w0 = -b * np.eye(3) *(x + 0*T * dx)

    #print( np.linalg.inv(Jx) @ (dx -  w0 + T * ddx)  )
    u = -a * np.linalg.inv(Jx) @ (dx -  w0 + 0*T * ddx ) + np.matrix([0,0,m*params.SysParams.g]).transpose()
    mean = [0, 0, 0]
    cov =  0*np.eye(3)
    x = np.random.multivariate_normal(mean, cov)

    u[0,0] = np.clip(u[0,0], -np.pi/4.0, np.pi/4.0)

    
    u[1,0] = np.clip(u[1,0], -np.pi/4.0, np.pi/4.0)
    u[2,0] = np.clip(u[2,0], 0, 1e9)

    return u 


def GNewtonController(x, dx, ddx, params, u):
    T = params.T
    a = params.a
    b = params.b
    m = params.SysParams.m
    JacobianFunc = params.J
    Jx = JacobianFunc(u, params.SysParams)
    
    w0 = -(b/T) * np.eye(3) *(x + T * dx)
    print( w0)
    H = dx @ dx.transpose()
    
    u = u - np.matrix([0,0,m*params.SysParams.g]).transpose()
    u = ( 1 *np.linalg.inv( Jx) @ (dx -  w0 + T * ddx)) + np.matrix([0,0,m*params.SysParams.g]).transpose()
    #u = -(a/T) * np.linalg.inv(Jx) @ (dx -  w0 + T * ddx) + np.matrix([0,0,m*params.SysParams.g]).transpose()

    mean = [0, 0, 0]
    cov =  0*np.eye(3)
    x = np.random.multivariate_normal(mean, cov)
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
m = 3.
g = 9.81
SysParams.addProperty("g", g)
SysParams.addProperty("m", m)


ContParams = SystemParameter()
#Known To be good
ContParams.addProperty("a", 5)
ContParams.addProperty("b", 3)
ContParams.addProperty("T", 0.01)
ContParams.addProperty("J", JacobianFunc)
ContParams.addProperty("SysParams", SysParams)


# ContParams.addProperty("a", 0.04)
# ContParams.addProperty("b", 0.01)
# ContParams.addProperty("T", 0.01)
# ContParams.addProperty("J", JacobianFunc)
# ContParams.addProperty("SysParams", SysParams)



PE = EulerMethod_Propogator(StateDim,Ts,T_end,FirstDerivFunction,SysParams,SecondDerivFunction,SysParams)

PE.InitialCondition_state  = np.matrix([2,-2,1]).transpose()
PE.InitialCondition_d_state  = np.matrix([0,0,0]).transpose()
PE.InitialCondition_dd_state  = np.matrix([0,0,0]).transpose()

PE.ControllerFunction = NewtonController
PE.ControllerParam = ContParams
PE.Ts_controller = 1/10

# PE.ControllerFunction = PID
# PE.ControllerParam = [10, 8, 6, 2, m]
# PE.Ts_controller = 1/10




u_in = np.matrix([[0],[0],[0]])
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

error = [np.linalg.norm(x) for x in PE.StateVector]
fig1, ax1 = plt.subplots()
ax1.plot(PE.TimeVect, error)

plt.show()

