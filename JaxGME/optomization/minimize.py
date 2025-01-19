import JaxGME
from JaxGME.backend import backend as bd
import jax
import numpy as np 
from scipy.optimize import minimize
import time
import json
import types

class Minimize(object):
    """
    Class that defines the optomization method that we will be using.
    """
    def __init__(self, x0, crystal, cost, mode=0, constraints='None', gmax=3.001, gmeParams={}, phcParams={}, tol=None):
        """
        Initalizes the class with all relevent general perameters

        Args:
            x0 : The inital perameters for hole potisition and radius
            crystal : The function defining the crystal we are optomizing on, should have input vars
            cost : the class object that defines cost function we are using
            mode : the mode we are optomizing on
            constraints : the class object that defines the types of constraints
            gmax : the gmax used in the gme computation
            gmeParams : the parameters to our GME simulation defined in dictionary
            phcParams : the parameters to our Photonic crystal definition, except for vars defined in dictionary
            tol : tolerance perameter needed for scipy.optomize.minimize. Default None
        """

        self.x0 = x0
        self.crystal = crystal
        self.cost = cost
        self.mode = mode
        self.constraints = constraints
        self.gmax = gmax
        self.gmeParams = gmeParams
        self.phcParams = phcParams
        self.tol = tol
        self.result = None
        

    def print_params(self):
        """Print the class parameters."""
        for key, value in self.__dict__.items():
            if key == 'crystal':
                print(f"{key}: {value.__name__}")
            elif key == 'cost':
                print('-----cost-----')
                self.cost.print_params()
                print('-----cost-----')
            else:
                print(f"{key}: {value}")

    def objective(self,vars):
        """
        defines the function that will be optomizaed over
        """
        phc = self.crystal(vars=vars,**self.phcParams)
        gme = JaxGME.GuidedModeExp(phc,self.gmax)
        gme.run(**self.gmeParams)
        out = self.cost.cost(gme,phc,self.mode)
        return(out)
    
    def scipy_objective(self,vars):
        """
        defines the scipy wrapper for the objective functions written in jax
        """
        return(np.array(self.objective(vars)))

    def minimize(self):
        """
        defines the optomization method that will be used
        """
        raise NotImplementedError("minimize() needs to be implemented by"
                                  "minimize subclasses")
    
    def save(self,file_path):
        """
        defines the function that saves the information about the class and 
        results of the optomization to a json file. It clears any file there
        and then writes 

        Args:
            file_path : the path the file should be stored at
        """

        #builds the dict recersivly
        def build_dict(data):
            for key, value in data.items():
                if isinstance(value,(bd.ndarray,list,np.ndarray)):
                    data[key] = bd.array(value).tolist()
                elif isinstance(value,types.FunctionType):
                    data[key] = value.__name__
                elif isinstance(value, (JaxGME.Cost)):
                    data[key] = build_dict(value.__dict__)
                elif isinstance(value, dict):
                    data[key] = build_dict(value)
                else:
                    data[key] = value
            return(data)
            
        save_data = build_dict(self.__dict__)

        try:
            with open(file_path, 'w') as json_file:
                json.dump(save_data, json_file, indent=4)
        except IOError as e:
            raise IOError(f"An error occurred while writing to {file_path}: {e}")


class BFGS(Minimize):
    """
    runs the BFGS minimization method from scipy.optomize.minimize
    """

    def __init__(self, x0, crystal, cost, disp=False, maxiter=None, gtol=1e-5, return_all=False, **kwargs):
        """
        Defines all of the inputs needed to run optomizations with BFGS optomization class.
        Default values are the defaults from scipy

        Args: 
            disp : Bool, set to True to print convergance messages
            maxiter : Int or None, The maximimum number of iterations that will be run
            gtol : Float, the tolerance on the gradent to stop optoimization
            return_all : Bool, returns all intermediate steps at the end of optimization
        """
        super().__init__(x0,crystal,cost,**kwargs)
        self.disp = disp
        self.maxiter = maxiter
        self.gtol = gtol
        self.return_all = return_all

    def minimize(self):
        """
        function to call to preform minimization
        """

        #define grad function
        gradFunc = jax.grad(self.objective)

        #scipy needs the gradient as a numpy array
        def scipy_grad(var):
            return(np.array(gradFunc(var)))
        
        #get the inital value to save
        self.inital_cost = self.objective(self.x0)
        
        #run optomization
        t1 = time.time()
        result = minimize(fun=self.scipy_objective,    
                          x0=self.x0,            
                          jac=scipy_grad, 
                          method='BFGS',
                          options={'disp': self.disp,
                                   'maxiter': self.maxiter,
                                   'gtol': self.gtol,
                                   'return_all': self.return_all}
                        )
        
        #save result
        self.result = dict(result.items())
        self.time = time.time()-t1




