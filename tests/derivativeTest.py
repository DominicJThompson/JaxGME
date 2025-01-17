#%%
import sys
import os

# Add the project root directory to sys.path
os.environ['JAX_ENABLE_X64'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import JaxGME
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
JaxGME.set_backend('jax')
import time

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,        # General font size
    'axes.titlesize': 18,   # Title font size
    'axes.labelsize': 16,   # Axis label font size
    'xtick.labelsize': 12,  # X-tick label font size
    'ytick.labelsize': 12,  # Y-tick label font size
    'legend.fontsize': 14   # Legend font size
})


def W1(vars=jnp.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

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
    return(vars)
# %%
def alphaCalc(holes):
    phc = W1(vars=holes)

    gme = JaxGME.GuidedModeExp(phc,gmax=2.00001)
    gme.run(kpoints=jnp.vstack([.8*jnp.pi]),numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)

    alpha = backCost.cost(gme,phc,20,0)

    return(alpha)

vars = W1Vars()
out = alphaCalc(vars)
gradFunc = jax.grad(alphaCalc)
grad = gradFunc(vars)

#%%
diff = 1e-5
t1 = time.time()
def finite_diff():
    out = alphaCalc(vars)
    finite_grad = jnp.zeros_like(grad)
    for i in range(grad.shape[0]):
        print(i)
        for j in range(grad.shape[1]):
            vars2 = W1Vars()
            vars2 = vars2.at[i,j].add(diff)
            finite_grad = finite_grad.at[i,j].set((alphaCalc(vars2)-out)/diff)
    return(finite_grad)
fGrad = finite_diff()
print(time.time()-t1)
print(grad)
print(fGrad)

# %%
plt.plot(grad.T)
plt.plot(fGrad.T,'--')
plt.show()
# %%
