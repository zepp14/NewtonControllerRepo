import rpyc
import numpy as np

from PhysicsEngine import EulerMethod_Propogator, hat, SystemParameter,diff1Function, diff2Function

class MyService(rpyc.Service):
    
    def __init__(self) -> None:
        self.changeTest =  1
        self.propogator = None
        self.client_np = None

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        
        
        print("Got a connection")
        pass
    
    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def get_answer(self): # this is an exposed method
        return 42

    def setNPLIB(self,npLib):
        self.client_np = npLib
        return 1

    def initProp_RigidBodyRot(self, Imat, T_s, Tend): # this is an exposed method

        Parameters = SystemParameter()
        Parameters.addProperty("Imat", Imat)
        self.propogator =  EulerMethod_Propogator(StateDim=3, Ts_world=T_s, T_end=Tend,
                                                 diff1_fun=diff1Function, diff1_args=Parameters,
                                                 diff2_fun=diff2Function,  diff2_args=Parameters)
        
        print("Propogator Enables - set initial conditions")
        return 

    def setInitialCond(self,x0,dx0,ddx0):
        self.propogator.InitialCondition_state = x0
        self.propogator.InitialCondition_d_state = dx0
        self.propogator.InitialCondition_dd_state = ddx0
        return print("Initial Conditions Set")


    def runFreeSim(self):
        u = np.matrix([0,0,0]).transpose()
        print("Run Free body sim")
        self.propogator.EulerLoop(u)
        print(self.propogator.StateVector)
        
        return 1



    def get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"



if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    server = ThreadedServer(MyService, port=18861, protocol_config={ 'allow_public_attrs': True,"allow_all_attrs":True, "allow_setattr":True,"allow_pickle":True})
    print("Server Starting on Port 18861")
    server.start()