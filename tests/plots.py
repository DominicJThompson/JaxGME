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
import json

def W1(vars=jnp.zeros((0,0)),NyChange=0,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.199):

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

def runGME(holes,path):
    phc = W1(vars=holes)

    gme = JaxGME.GuidedModeExp(phc,gmax=4.0000001)
    gme.run(kpoints=path,numeig=30,compute_im=False,verbose=False)

    return(gme,phc)

#%%
#------------- band diagram ------------
gme,phc = runGME(jnp.zeros((0,0)),jnp.vstack((jnp.linspace(.5*jnp.pi,jnp.pi,50),jnp.zeros(50))))

# %%
plt.plot(gme.kpoints[0,:]/2/jnp.pi,gme.freqs, color='darkviolet')
plt.plot(gme.kpoints[0,:]/2/jnp.pi,gme.freqs[:,1], color='darkviolet')
plt.plot()

plt.ylim()
plt.xlim(.25,.5)
plt.xlabel(r'$k_x/2\pi$')
plt.ylabel(r'$\omega a/2\pi c$')


# %%
