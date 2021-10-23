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
    
        




#Propogator Class
class RungeKutta4_Propogator(object):
    def __init__(self, StateDim, Ts_world, T_end, diff1_fun, diff1_args, diff2_fun=None,  diff2_args=None):

        self.StateDim = StateDim
        self.Ts_world = Ts_world
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


    def EvalDiff2RK(self, x0, dx0, ddx0, u_in, t):
        h = t
        
        func1 = self.FirstDeriv
        func2 = self.SecondDeriv
        args = self.SecondDeriv_args

        ds_a = dx0 + 0 * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0)
        s_a =  x0 + 0*ds_a + (1/2)*(0**2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0)
        k1 = func2(u_in , params=args, State=s_a, dState=ds_a, ddState=ddx0)


        ds_b = dx0 + (h/2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k1*(h/2))
        s_b =  x0 + (h/2)*ds_b + (1/2)*((h/2)**2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k1*(h/2))
        k2 = func2(u_in , params=args, State=s_b, dState=ds_b, ddState=(ddx0+k1*(h/2)))

        ds_c = dx0 + (h/2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k2*(h/2))
        s_c =  x0 + (h/2)*ds_c + (1/2)*((h/2)**2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k2*(h/2))
        k3 = func2(u_in , params=args, State=s_c, dState=ds_c, ddState=(ddx0+k2*(h/2)))
        
        ds_d = dx0 + (h) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k3*(h))
        s_d =  x0 + (h)*ds_d + (1/2)*((h)**2) * func2 (u_in , params=args, State=x0, dState=dx0, ddState=ddx0+k3*(h))
        k4 = func2(u_in , params=args, State=x0, dState=ds_c, ddState=(ddx0+k3*(h)))

        out = ddx0 + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
        
        
        return out

    def EvalDiff1RK(self, x0, dx0, ddx0, u_in, t):
        h = t
        
        func1 = self.FirstDeriv
        func2 = self.SecondDeriv
        args = self.SecondDeriv_args

        dds_a = self.EvalDiff2RK(x0, dx0, ddx0, u_in, t=0)
        s_a =  x0 + 0*dx0 + (1/2)*(0**2) * dds_a 
        k1 = func1(u_in , params=args, State=s_a, dState=dx0, ddState=dds_a)

        dds_b = self.EvalDiff2RK(x0, (dx0+k1*(h/2)), ddx0, u_in, t=(h/2))
        s_b =  x0 + (h/2)*(dx0+k1*(h/2))+ (1/2)*((h/2)**2) * dds_b 
        k2 = func1(u_in , params=args, State=s_b, dState=(dx0+k1*(h/2)), ddState=dds_b)

        dds_c = self.EvalDiff2RK(x0, (dx0+k2*(h/2)), ddx0, u_in, t=(h/2))
        #dds_c = dds_b
        s_c =  x0 + (h/2)*(dx0+k2*(h/2))+ (1/2)*((h/2)**2) * dds_c 
        k3 = func1(u_in , params=args, State=s_c, dState=(dx0+k2*(h/2)), ddState=dds_c)
        
        dds_d = self.EvalDiff2RK(x0, (dx0+k3*(h)), ddx0, u_in, t=(h))
        #dds_d = 2*dds_c
        s_d =  x0 + (h)*(dx0+k3*(h))+ (1/2)*((h)**2) * dds_d 
        k4= func1(u_in , params=args, State=s_d, dState=(dx0+k3*(h)), ddState=dds_d)

        out = dx0 + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)


        
        return out

    def EvalStateRK(self, x0, dx0, ddx0, u_in, t):
        h = t
        

        dds_a = self.EvalDiff2RK(x0, dx0, ddx0, u_in, t = 0)
        ds_a =  self.EvalDiff1RK( x0, dx0, ddx0, u_in, t =0)
        k1 = x0 + (0)*ds_a + (1/2)*((0)**2) * dds_a

        dds_b = self.EvalDiff2RK((x0+k1*(h/2)), dx0, ddx0, u_in, t = h/2)
        ds_b =  self.EvalDiff1RK((x0+k1*(h/2)), dx0, ddx0, u_in, t = h/2)
        k2 = (x0 + k1*(h/2)) + ((h/2))*ds_b + (1/2)*(((h/2))**2) * dds_b

        dds_c = self.EvalDiff2RK((x0+k2*(h/2)), dx0, ddx0, u_in, t = h/2)
        ds_c =  self.EvalDiff1RK((x0+k2*(h/2)), dx0, ddx0, u_in, t = h/2)
        k3 = (x0 + k2*(h/2)) + ((h/2))*ds_c + (1/2)*(((h/2))**2) * dds_c

        dds_d = self.EvalDiff2RK((x0+k3*(h)), dx0, ddx0, u_in, t = h)
        ds_d =  self.EvalDiff1RK((x0+k3*(h)), dx0, ddx0, u_in, t = h)
        k4 = (x0 + k3*(h)) + ((h))*ds_c + (1/2)*(((h))**2) * dds_c
        

        out = dx0 + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
    
        
        return out

    def EvalStateRK_Simple(self, x0, dx0, ddx0, u_in, t):
        h = t

        dds_a = ddx0
        ds_a =  self.EvalDiff1RK( x0, dx0, ddx0, u_in, t =0)
        k1 = x0 + (0)*ds_a + (1/2)*((0)**2) * dds_a

        dds_b = ddx0
        ds_b =  self.EvalDiff1RK((x0+k1*(h/2)), dx0, ddx0, u_in, t = h/2)
        k2 = (x0 + k1*(h/2)) + ((h/2))*ds_b + (1/2)*(((h/2))**2) * dds_b

        dds_c = ddx0
        ds_c =  self.EvalDiff1RK((x0+k2*(h/2)), dx0, ddx0, u_in, t = h/2)
        k3 = (x0 + k2*(h/2)) + ((h/2))*ds_c + (1/2)*(((h/2))**2) * dds_c

        dds_d = ddx0
        ds_d =  self.EvalDiff1RK((x0+k3*(h)), dx0, ddx0, u_in, t = h)
        k4 = (x0 + k3*(h)) + ((h))*ds_d + (1/2)*(((h))**2) * dds_d
        

        out = dx0 + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
    
        
        return out

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


    def RK4_loop(self, u_in):
        
        self.StateVector = []
        self.dStateVector = []
        self.ddStateVector = []

        x =  self.InitialCondition_state
        dx =  self.InitialCondition_d_state
        ddx =  self.InitialCondition_dd_state
        t = self.Ts_world

        for Ts in self.TimeVect:
            print(Ts)
            ddx_new = self.EvalDiff2RK(x, dx, ddx, u_in, t)
            dx_new = self.EvalDiff1RK(x, dx, ddx, u_in, t)
            x_new = self.EvalStateRK_Simple(x, dx, ddx, u_in, t)
            ddx = ddx_new
            dx = dx_new
            x =  x_new
            self.storeResults(x, dx, ddx)


        self.StateVector = np.array(self.StateVector)
        self.dStateVector = np.array(self.dStateVector)
        self.ddStateVector = np.array(self.ddStateVector)
        #k1 = func(u_in , params=args, State=x0, dState=func, ddState=ddx0)


        #print(k1)
        #self.StateVector = 

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
    ddState = ImatInv @ ( uVec  - np.cross (dStateVec, (Imat @ dStateVec)))

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
    
    