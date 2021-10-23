import numpy as np
import matplotlib.pyplot as plt
import time
from PhysicsEngine import EulerMethod_Propogator, SystemParameter, hat, diff1Function, diff2Function

StateDim = 3
Ts = 1/3000
T_end = 10

#Simple controller
def PID(x, dx, ddx, params):
    kp = params[0]
    kd = params[1]

    u = -(kp * x + kd*dx)
    return u

#Newton controller
def RigidBodyNewton(x, dx, ddx, params):
    Im = params.Imat
    a = params.a
    T = params.T
    u = -(a/T) * np.linalg.inv(Im) @ (dx + T*ddx)
    return u 


Inonsym = np.matrix([[1, 0.002, -0.002],[0,0.4, -0.001],[0, 0, 0.3]])
Imat = 0.5 * (Inonsym.transpose() + Inonsym)


Parameters = SystemParameter()
Parameters.addProperty("Imat", Imat)

ContParam = SystemParameter()
ContParam.addProperty("Imat", Imat)
ContParam.addProperty("a", 0.01)
ContParam.addProperty("T",0.01)

u_in = np.matrix([[0],[0],[0]])

PE = EulerMethod_Propogator(StateDim, Ts, T_end,
                            diff1_fun=diff1Function, diff1_args=Parameters,
                            diff2_fun=diff2Function,  diff2_args=Parameters)


PE.InitialCondition_state  = np.matrix([0,0,0]).transpose()
PE.InitialCondition_d_state  = np.matrix([1,-2,2]).transpose()
PE.InitialCondition_dd_state  = np.matrix([0,0,0]).transpose()

PE.ControllerFunction = PID
PE.ControllerParam = [1,1]
PE.Ts_controller = 1/10

PE.ControllerFunction = RigidBodyNewton
PE.ControllerParam = ContParam
PE.Ts_controller = 1/100

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


h =  [ np.linalg.norm(Imat @ w.transpose()) for w in PE.dStateVector]
fig1, ax1 = plt.subplots()
ax1.plot(PE.TimeVect,h )


plt.show()