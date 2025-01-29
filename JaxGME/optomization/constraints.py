from JaxGME.backend import backend as bd
import jax
import JaxGME


class ConstraintManager(object):
    """
    The class that defines constraints to be placed on the optomization.
    Each function of the class is a wrapper that defines another function
    """
    def __init__(self,x0=[],numberHoles=0,**kwargs):
        """
        Initalize the relevent fields

        Args:
            x0: inital variables
            numberHoles: number of holes on each side being changed
        """

        #this contains the constraints to be easily converted to scipy constraints:
        #{'name': {'type': 'ineq'/'eq', 'fun': constraint function, 'args': (tuple of args)},..}
        #the idea is to use list(self.constraints.values()) later to get scipy constraints
        self.constraints = {}

        #this contains the name and discription of the constraint:
        #{'name': {'discription': short discription, 'args': any relevent arguments}},..}
        #this is for saving purposes
        self.constraintsDisc = {}

        #defualt arguments from optimizer and initialized that need to be set
        self.defaultArgs = {'x0':x0,
                            'numberHoles':numberHoles,
                            **kwargs
                            }

    def remove_constraint(self, name):
        """
        Remove a constraint by name.
        """
        if name in self.constraints:
            del self.constraints[name]
    
    def update_args(self, name, args):
        """
        Update the arguments of a specific constraint.
        """
        if name not in self.constraints:
            raise ValueError(f"Constraint {name} does not exist.")
        func = self.defaultArgs.get(name, (None,))[0]
        self.constraints[name]['fun'] = self._wrap_function(func, args)
        self.defaultArgs[name] = args

    def get_active_constraints(self):
        """
        Get a list of active constraints for optimization, formatted for SciPy.
        """
        return list(self.constraints.values())
    
    def _wrap_function(self, func, args):
        """
        Wrap the constraint function to pass updated arguments dynamically.
        """
        def wrapped(x):
            return float(func(x, *args))
        return wrapped
    
    def _wrap_grad(self, func, args):
        """
        Wraps the gradient of the constraints
        """
        def wrapped(x):
            return func(x,*args).tolist()
        return wrapped

    #------------default constraints to add-------------
        
    def add_inside_unit_cell(self, name):
        """
        Keep x and y values bound in box of [-.5,.5], assume
        Assume xs of shape [*xs,*ys,*rs]
        """
        for i in range((2*self.defaultArgs['numberHoles'])):
            #for each x value
            self.constraints[name+'_x_'+str(i)] = {
                'type':'ineq',
                'fun': self._wrap_function(self._inside_unit_cell,(i,)),
                'jac': self._wrap_grad(jax.grad(self._inside_unit_cell),(i,))
            }
            #for each y value
            self.constraints[name+'_y_'+str(i)] = {
                'type':'ineq',
                'fun': self._wrap_function(self._inside_unit_cell,(i+2*self.defaultArgs['numberHoles'],)),
                'jac': self._wrap_grad(jax.grad(self._inside_unit_cell),(i+2*self.defaultArgs['numberHoles'],))
            }

        self.constraintsDisc[name] = {
            'discription': """Keeps the x and y values bound in [-.5,.5] so they stay in the unit cell""",
            'args':{}
        }
    
    def add_min_rad(self,name,minRad):
        """
        Enforces a minimum radius that the holes may not go below
        Assume xs of shape [*xs,*ys,*rs]
        """
        for i in range(self.defaultArgs['numberHoles']*2):
            #for each radius
            self.constraints[name+str(i)] = {
                'type': 'ineq',
                'fun': self._wrap_function(self._min_rad,(i+self.defaultArgs['numberHoles']*4,minRad,)),
                'jac': self._wrap_grad(jax.grad(self._min_rad),(i+self.defaultArgs['numberHoles']*4,minRad,))
            }
        self.constraintsDisc[name] = {
            'discription':  """Enforces the minimum radius""",
            'args': {'minRad': minRad}
        }
    
    def add_max_rad(self,name,maxRad):
        """
        Enforces a maximum radius that the holes may not go above
        Assume xs of shape [*xs,*ys,*rs]
        """
        for i in range(self.defaultArgs['numberHoles']*2):
            #for each radius
            self.constraints[name+str(i)] = {
                'type': 'ineq',
                'fun': self._wrap_function(self._max_rad,(i+self.defaultArgs['numberHoles']*4,maxRad,)),
                'jac': self._wrap_grad(jax.grad(self._max_rad),(i+self.defaultArgs['numberHoles']*4,maxRad,))
            }
        self.constraintsDisc[name] = {
            'discription':  """Enforces the maximum radius""",
            'args': {'maxRad': maxRad}
        }
    
    def add_min_dist(self,name,minDist,buffer,varsPadded):
        """
        Enforces a minimum distance between holes including the radius. 
        Enforce this distance for a buffer number of holes on each side
        Need the vars padded to get aditional positional information from holes we are not optomizing over
        Assume xs of shape [*xs,*ys,*rs], also assumes holes are ordered in the order they appear

        Args: 
            buffer: the numbers of holes on either side that we enforce the distance 
            varsPadded: inital variables with padding added on either side for additinoal holes, 
                        should be 2*buffer larger then vars for each variable
        """
        for i in range(self.defaultArgs['numberHoles']*2+buffer):
            for j in range(buffer):
                if i+j+1<buffer: #if hole is looking at another hole that doesn't move
                    continue
                self.constraints[name+str(i)+'_'+str(j+1)] = {
                    'type': 'ineq',
                    'fun': self._wrap_function(self._min_dist,(minDist,i,j+1,buffer,varsPadded)),
                    'jac': self._wrap_grad(jax.grad(self._min_dist),(minDist,i,j+1,buffer,varsPadded))
                }
        self.constraintsDisc[name] = {
            'discription': """Enforces a minimum radius between the holes within a buffer number of holes""",
            'args': {'minDist': minDist, 'buffer': buffer}
        }

    def add_freq_bound(self,name,minFreq,maxFreq):
        """
        Enforces the frequency beging bound within the region [minFreq,maxFreq]

        Args:
            minFreq: minimum alowable frequency
            maxFreq: maximum alowable frequency
        """
        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function(self._freq_bound,(minFreq,maxFreq,)),
            'jac': self._wrap_grad(jax.grad(self._freq_bound),(minFreq,maxFreq,))
        }
        self.constraintsDisc[name] = {
            'discription': """Enforces the frequency to be within the given frequency range""",
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq}
        }


    
    #----------functions that define default constraints----------
        
    def _inside_unit_cell(self,x,i):
        return(-bd.abs(x[i]-self.defaultArgs['x0'][i]-.5))
    
    def _min_rad(self,x,i,minRad):
        return(minRad-x[i])
    
    def _max_rad(self,x,i,maxRad):
        return(x[i]-maxRad)
    
    def _min_dist(self,x,minDist,i,j,buffer,varsPadded):
        #i indicates the hole to look at
        #j indicates the number of holes about i we should look

        numH = self.defaultArgs['numberHoles']
        if i<buffer:
            #we are looking at hole that is in the padding and comparing to variable hole
            xi, yi, ri = varsPadded[i], varsPadded[i+(numH+buffer)*2], varsPadded[i+4*(numH+buffer)]
            xj, yj, rj = x[i+j-buffer], x[i+j-buffer+2*numH], x[i+j-buffer+numH*4]
        
        elif i>=numH*2 and i+j>=numH*2+buffer:
            #we are looking at the last few changing holes
            xi,yi,ri = x[i-buffer], x[i-buffer+2*numH], x[i-buffer+numH*4]
            xj,yj,rj = varsPadded[i+j], varsPadded[i+j+(numH+buffer)*2], varsPadded[i+j+4*(numH+buffer)]

        else:
            #we ar fully inside the region where we are changing holes
            xi,yi,ri = x[i-buffer], x[i-buffer+2*numH], x[i-buffer+numH*4]
            xj,yj,rj = x[i-buffer+j], x[i-buffer+2*numH+j], x[i-buffer+numH*4+j]

        #distance between the holes
        dist = bd.sqrt((xi-xj)**2+(yi-yj)**2)-rj-ri

        return(minDist-dist)

    def _freq_bound(self,x,minFreq,maxFreq):

        #calculates the gme and then ensure the frequency is between the two values

        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = JaxGME.GuidedModeExp(phc,self.defaultArgs['gmax'])
        gme.run(**self.defaultArgs['gmeParams'])
        
        freq = gme.freqs[0,self.defaultArgs['mode']]

        return(bd.abs(freq-(minFreq+maxFreq)/2)-(maxFreq-minFreq)/2)



