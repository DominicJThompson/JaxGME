#%%
import sys
import os

# Add the project root directory to sys.path
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import JaxGME
from JaxGME import ConstraintManager
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
JaxGME.set_backend('jax')
import time
import multiprocessing as mp
import json


def W1(vars=jnp.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    vars = vars.reshape((3,NyChange*2))

    lattice = JaxGME.Lattice(jnp.array([1,0]),jnp.array([0,Ny*jnp.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = JaxGME.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(JaxGME.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    for i in range(Ny*2-1):
        iy = i-Ny
        if i>=Ny:
            iy+=1

        if jnp.abs(iy)<=NyChange:
            continue
        
        #move the hole over by half a unit cell if they are on odd rows
        if iy%2==1:
            x = .5
        else:
            x = 0

        #the y component should be scaled by the factor of np.sqrt(3)/2
        y = iy*jnp.sqrt(3)/2

        #now we can add a circle with the given positions
        phc.add_shape(JaxGME.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

def W1Vars(NyChange=3,ra=.3,key=0):
    key = jax.random.PRNGKey(key)
    vars = jnp.zeros((3,NyChange*2))
    vars = vars.at[2,:].set(ra)
    for i in range(NyChange*2):
        iy = i-NyChange
        if i>=NyChange:
            iy+=1
        if iy%2==1: 
            vars = vars.at[0,i].set(.5)
        vars = vars.at[1,i].set(iy*jnp.sqrt(3)/2)

    vars = vars.flatten()

    #generate some random noise for inital value
    noise = jax.random.normal(key,shape=vars.shape)*(1/266)

    return(vars+noise)


def worker_function(input):

    #make directory to save files
    with open(input['path'], "w") as f:
        pass

    cost = JaxGME.Backscatter()
    ks = jnp.linspace(jnp.pi*.5,jnp.pi,25)

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':jnp.array([[int(ks[8])],[0]])}
    vars = W1Vars(key=input['key'])
    
    #define constraints
    manager = ConstraintManager(x0=vars,
                                numberHoles=3,
                                crystal=W1,
                                phcParams={},
                                gmeParams=gmeParams,
                                gmax=3.01,
                                mode=20)
    manager.add_inside_unit_cell('Inside',.5)
    manager.add_rad_bound('minimumRadius',27.5/266,.4)
    manager.add_min_dist('minDist',40/266,3,W1Vars(NyChange=3+3))
    manager.add_gme_constrs('gme_constrs',minFreq=.26,maxFreq=.28,minNg=6.8,maxNg=6.9,ksBefore=[float(ks[4]),float(ks[6])],ksAfter=[float(ks[14]),float(ks[20])],bandwidth=.005,slope='down')
    
    #run minimization
    minim = JaxGME.TrustConstr(vars,W1,cost,mode=20,maxiter=500,gmeParams=gmeParams,constraints=manager,path=input['path'])
    minim.minimize()
    minim.save(input['path'])

if __name__=="__main__":
    input = {'path':f"tests/media/opt{int(sys.argv[1])}.json",'key':int(sys.argv[1])}
    worker_function(input)  # Compute the result

# %%
