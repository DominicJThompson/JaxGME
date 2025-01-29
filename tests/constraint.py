#%%
import sys
import os

# Add the project root directory to sys.path
os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import JaxGME
import jax
from JaxGME import ConstraintManager
import matplotlib.pyplot as plt
import jax.numpy as jnp
JaxGME.set_backend('jax')
import time

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

def W1Vars(NyChange=3,ra=.3):
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
    return(vars)
#%%
vars = W1Vars()

#%%
gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':jnp.array([[jnp.pi*.75],[0]])}
phcParams = {}

manager = ConstraintManager(x0=vars,
                            numberHoles=3,
                            crystal=W1,
                            phcParams=phcParams,
                            gmeParams=gmeParams,
                            gmax=4,
                            mode=20)

manager.add_inside_unit_cell('Inside')
manager.add_min_rad('minimumRadius',.2)
manager.add_min_dist('minDist',.1,3,W1Vars(NyChange=3+3))
manager.add_freq_bound('freqBound',.1,1)

# %%
name = 'freqBound'
#print(manager.constraints[name]['fun'](vars))
grad = manager.constraints[name]['jac'](vars)
#%%
manager.constraints
# %%
diff = 1e-5
t1 = time.time()
def finite_diff():
    out = manager.constraints[name]['fun'](vars)
    finite_grad = jnp.zeros(18)
    for i in range(18):
        print(i)
        vars2 = W1Vars()
        vars2 = vars2.at[i].add(diff)
        finite_grad = finite_grad.at[i].set((manager.constraints[name]['fun'](vars2)-out)/diff)
    return(finite_grad)
fGrad = finite_diff()
print(time.time()-t1)
print(grad)
print(fGrad)
# %%
plt.plot(grad)
plt.plot(fGrad,'--')
# %%
