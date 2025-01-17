

class Minimize(object):
    """
    Class that defines the optomization method that we will be using.
    """

    def __init__(self, initialParams, crystal, cost, constraints='None', gmeParams={}, phcParams={}):
        """
        Initalizes the class with all relevent general perameters

        Args:
            initialParams : The inital perameters for hole potisition and radius
            crystal : The function defining the crystal we are optomizing on, should have input vars
            cost : the class object that defines cost function we are using
            constraints : the class object that defines the types of constraints
            gmeParams : the parameters to our GME simulation defined in dictionary
            phcParams : the parameters to our Photonic crystal definition, except for vars defined in dictionary
        """

        self.initialParams = initialParams
        self.crystal = crystal
        self.cost = cost
        self.constraints = constraints
        self.gmeParams = gmeParams
        self.phcParams = phcParams

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

