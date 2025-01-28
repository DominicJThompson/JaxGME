from JaxGME.backend import backend as bd

class ConstraintManager(object):
    """
    The class that defines constraints to be placed on the optomization.
    Each function of the class is a wrapper that defines another function
    """
    def __init__(self,x0,numberHoles=3):
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

        #this contains expensive computations from the objective function to speed up compuation
        #{'name':computation,..}
        self.cashe = {}

        #number of holes begin changed, twice the number on each side
        self.numberHoles = numberHoles*2

        #initaial variables
        self.x0 = x0

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
        func = self.default_args.get(name, (None,))[0]
        self.constraints[name]['fun'] = self._wrap_function(func, args)
        self.default_args[name] = args

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
            return func(x, *args)
        return wrapped

    def update_cache(self, key, value):
        """
        Update the shared cache with a key-value pair.
        """
        self.cache[key] = value

    #------------default constraints to add-------------
        
    def inside_unit_cell(self, name):
        """
        Keep x and y values bound in box of [-.5,.5], assume
        Assume xs of shape [*xs,*ys,*rs]
        """
        for i in range((2*self.numberHoles)):
            #for each value that isnt a radius
            self.constraints[name+str(i)] = {
                'type':'ineq',
                'fun': self._wrap_function(self._inside_unit_cell,(i,))
            }
        self.constraintsDisc[name] = {
            'discription': """Keeps the x and y values bound in [-.5,.5] so they stay in the unit cell""",
            'args':{}
        }
    
    def min_rad(self,name,minRad):
        """
        enforces a minimum radius that the holes may not go below
        """
        for i in range(self.numberHoles):
            #for each radius
            self.constraints[name+str(i)] = {
                'type': 'ineq',
                'fun': self._wrap_function(self._min_rad,(i+self.numberHoles*2,minRad,))
            }
        self.constraintsDisc[name] = {
            'discription':  """Enforces the minimum radius that is taken as an arg""",
            'args': {'minRad': minRad}
        }
    
    #----------functions that define default constraints----------
        
    def _inside_unit_cell(self,x,i):
        return(-bd.abs(x[i]-self.x0[i]-.5))
    
    def _min_rad(self,x,i,minRad):
        return(minRad-x[i])


    