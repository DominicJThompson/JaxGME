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

gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':jnp.array([[jnp.pi*.75],[0]])}
phcParams = {}

manager = ConstraintManager(x0=vars,
                            numberHoles=3,
                            crystal=W1,
                            phcParams=phcParams,
                            gmeParams=gmeParams,
                            gmax=4,
                            mode=20)

#manager.add_inside_unit_cell('Inside')
#manager.add_min_rad('minimumRadius',.2)
#manager.add_min_dist('minDist',.1,3,W1Vars(NyChange=3+3))
#manager.add_freq_bound('freqBound',.1,1)
#manager.add_ng_bound('ngBound',1,10)
#manager.add_monotonic_band('monotomicBand',[jnp.pi*.5],[jnp.pi],'down')
#manager.add_bandwidth('bandwidth',[jnp.pi*.5],[jnp.pi],.1)
#manager.add_ng_others('ngOthers',[jnp.pi*.5],[jnp.pi],'down')
manager.add_gme_constrs('gme_constrs',minFreq=.1,maxFreq=1,minNg=10,maxNg=20,ksBefore=[jnp.pi*.5],ksAfter=[jnp.pi],bandwidth=.01,slope='down')
manager.constraintsDisc
#%%
name = 'gme_constrs'
print(manager.constraints[name]['fun'](vars))
grad = manager.constraints[name]['jac'](vars)
#%%
print(grad)
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
def finite_difference_jacobian(f, x, eps=1e-5):
    """
    Compute the Jacobian of function f at x using forward differences.
    
    Parameters:
    - f: Function that takes a vector x and returns a vector (shape (m,))
    - x: Point at which to evaluate the Jacobian (shape (n,))
    - eps: Small step size for finite differences
    
    Returns:
    - J: Jacobian matrix of shape (m, n)
    """
    x = jnp.asarray(x, dtype=float)
    m = len(f(x))  # Number of outputs
    n = len(x)  # Number of inputs
    J = jnp.zeros((m, n))
    f_x = f(x)

    for i in range(n):
        x_forward = x.at[i].add(eps)
        f_forward = f(x_forward)
        
        J = J.at[:, i].set((jnp.array(f_forward) - jnp.array(f_x)) / eps)  # Forward difference
    
    return J
# %%
fGrad = finite_difference_jacobian(manager.constraints[name]['fun'], vars, eps=1e-5)
# %%
plt.plot(jnp.abs(fGrad.T))
plt.plot(jnp.abs(jnp.array(grad).T),'--')
plt.ylim(0,.1)
# %%
plt.imshow(grad)
# %%
