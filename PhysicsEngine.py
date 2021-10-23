import numpy as np
from numpy.lib.index_tricks import diag_indices
import scipy
import struct

#Usfull Hat Func
def hat(vec):

    vecSkew = np.matrix([[  0,     -vec[2,0],  vec[1,0]],
                        [vec[2,0],   0    , -vec[0,0]],
                        [-vec[1,0],  vec[0,0],   0    ]])
    return vecSkew


class EulerMethod_Propogator(object):
    def __init__(self, StateDim, Ts_world, T_end, diff1_fun, diff1_args, diff2_fun=None,  diff2_args=None):
        self.StateDim = StateDim
        self.Ts_world = Ts_world
        self.ControllerFunction = None
        self.ControllerParam = None
        self.Ts_controller = None
        #standard Function Input:
        # func(u_in , params, State, dState, ddState)
        self.FirstDeriv = diff1_fun
        self.FirstDeriv_args =  diff1_args

        self.SecondDeriv = diff2_fun
        self.SecondDeriv_args =  diff2_args
        TimeVect = [] 
        TCond = 0
        while TCond < T_end:
            TCond = TCond + Ts_world
            TimeVect.append(TCond)
        self.TimeVect = np.array(TimeVect)

        self.StateVector = []
        self.dStateVector = []
        self.ddStateVector = []
        
        self.InitialCondition_state = np.asmatrix(np.zeros((StateDim,1)))
        self.InitialCondition_d_state = np.asmatrix(np.zeros((StateDim,1)))
        self.InitialCondition_dd_state = np.asmatrix(np.zeros((StateDim,1)))

    def OneStep(self,u_in, x0, dx0, ddx0, t):

        func1 = self.FirstDeriv
        func2 = self.SecondDeriv
        args = self.SecondDeriv_args

        ddx = func2(u_in , args, x0, dx0, ddx0)
        dx =  dx0 + t * ddx0
        x  =  x0 + (t * dx0) 
        

        return x, dx, ddx

    def storeResults(self, x, dx, ddx):

        xArr = []
        dxArr = []
        ddxArr  = []

        for i in range(0,self.StateDim):
            xArr.append(x[i,0])
            dxArr.append(dx[i,0])
            ddxArr.append(ddx[i,0])


        self.StateVector.append(xArr)
        self.dStateVector.append(dxArr)
        self.ddStateVector.append(ddxArr)
        

    def EulerLoop(self, u_in ):

        self.StateVector = []
        self.dStateVector = []
        self.ddStateVector = []

        x =  self.InitialCondition_state
        dx =  self.InitialCondition_d_state
        ddx =  self.InitialCondition_dd_state
        t = self.Ts_world
        for Ts in self.TimeVect:
            
            x, dx, ddx = self.OneStep(u_in, x, dx, ddx, t)
            self.storeResults(x, dx, ddx)

    
        self.StateVector = np.array(self.StateVector)
        self.dStateVector = np.array(self.dStateVector)
        self.ddStateVector = np.array(self.ddStateVector)
    
        
    def EulerLoop_w_Controller(self):

        self.StateVector = []
        self.dStateVector = []
        self.ddStateVector = []

        x =  self.InitialCondition_state
        dx =  self.InitialCondition_d_state
        ddx =  self.InitialCondition_dd_state
        u_in =  np.asmatrix(np.zeros((self.StateDim,1)))
        t = self.Ts_world
        Tc_last = 0
        Tsc = self.Ts_controller
        for Ts in self.TimeVect:
            if(abs(Ts - Tc_last) > Tsc):
                Tc_last = Ts
                u_in = self.ControllerFunction(x, dx, ddx, self.ControllerParam)

            x, dx, ddx = self.OneStep(u_in, x, dx, ddx, t)
            self.storeResults(x, dx, ddx)

    
        self.StateVector = np.array(self.StateVector)
        self.dStateVector = np.array(self.dStateVector)
        self.ddStateVector = np.array(self.ddStateVector)





#systemParameter class
class SystemParameter(object):
    def __init__(self):
        self.NullProp = None
        
    
    def addProperty(self,StrName, Value):
        self.__dict__[StrName] = Value

        

#DEFINE DERIVATIVES

   

def diff2Function(u_in , params, State, dState, ddState):
    
    Imat = params.Imat
    ImatInv = np.linalg.inv(Imat)
    #ImatInv =Imat
    uVec = u_in 
    dStateVec = dState
    #hat(dStateVec)
    ddState = ImatInv @ ( uVec  - hat(dStateVec)@(Imat @ dStateVec))

    return ddState
    
def diff1Function(u_in , params, State, dState, ddState, ts=0):

    #dStateK1 = dState + ts * diff2Function(u_in , params, State, dState, ddState)
    dStateK1 = dState
    return dStateK1


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    StateDim = 3
    Ts = 1/2000
    T_end = 15

    #Define Param Struct
    Inonsym = np.matrix([[1, 0.002, -0.002],[0,0.4, -0.001],[0, 0, 0.3]])
    #Inonsym = np.matrix([[3, 0.02, 0],[0,1, 0],[0, 0, 0.3]])
    Imat = 0.5 * (Inonsym.transpose() + Inonsym)
    
    #Imat = np.eye(3)
    #print(np.linalg.eigvals(Imat))
    Parameters = SystemParameter()
    Parameters.addProperty("Imat", Imat)


    u_in = np.matrix([[0],[0],[0]])
    
    
    

    PE = EulerMethod_Propogator(StateDim, Ts, T_end,
                                diff1_fun=diff1Function, diff1_args=Parameters,
                                diff2_fun=diff2Function,  diff2_args=Parameters)


    PE_RK = RungeKutta4_Propogator(StateDim, Ts, T_end,
                                    diff1_fun=diff1Function, diff1_args=Parameters,
                                    diff2_fun=diff2Function,  diff2_args=Parameters)

    PE.InitialCondition_state  = np.matrix([0,0,0]).transpose()
    PE.InitialCondition_d_state  = np.matrix([1,-2,0]).transpose()
    PE.InitialCondition_dd_state  = np.matrix([0,0,0]).transpose()

    # PE_RK.InitialCondition_state  = PE.InitialCondition_state
    # PE_RK.InitialCondition_d_state  = PE.InitialCondition_d_state
    # PE_RK.InitialCondition_dd_state  = PE.InitialCondition_dd_state

    startT =  time.time()
    PE.EulerLoop(u_in)
    #PE_RK.RK4_loop(u_in)
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
    
    