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


def W1(vars=jnp.zeros((0,0)),NyChange=0,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

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

def varRad(ra):
    path = jnp.vstack((jnp.array([2.308,2.308+1e-3]),jnp.zeros(50)))

    phc = W1(ra=ra)
    gme = JaxGME.GuidedModeExp(phc,gmax=4.0000000001)
    gme.run(kpoints=path,numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)

    alpha = backCost.cost(gme,phc,20)
    ng = (1/(2*jnp.pi))*jnp.abs((gme.kpoints[0,1]-gme.kpoints[0,0])/(gme.freqs[1,20]-gme.freqs[0,20]))
    return([alpha,ng])

def varRadAndK(ra):
    path = jnp.vstack((jnp.linspace(jnp.pi*.5,jnp.pi,51),jnp.zeros(51)))

    phc = W1(ra=ra)
    gme = JaxGME.GuidedModeExp(phc,gmax=4.0000000001)
    gme.run(kpoints=path,numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)
    alpha = []
    for i in range(50):
        alpha.append(backCost.cost(gme,phc,20,i))
    alpha = jnp.array(alpha)
    ng = (1/(2*jnp.pi))*jnp.abs((gme.kpoints[0,1:]-gme.kpoints[0,:-1])/(gme.freqs[1:,20]-gme.freqs[:-1,20]))
    return(jnp.vstack((alpha,ng)))

def varNy(Ny):
    path = jnp.vstack((jnp.array([2.308,2.308+1e-5]),jnp.zeros(2)))
    phc = W1(Ny=Ny)
    gme = JaxGME.GuidedModeExp(phc,gmax=4.0000000001)
    gme.run(kpoints=path,numeig=Ny*2+1,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)
    n = Ny*2
    alpha = backCost.cost(gme,phc,n)
    ng = (1/(2*jnp.pi))*jnp.abs((gme.kpoints[0,1]-gme.kpoints[0,0])/(gme.freqs[1,n]-gme.freqs[0,n]))
    return([alpha,gme.freqs[0,n],ng])

def varK(k):
    phc = W1()
    path = jnp.vstack((jnp.array([k,k+1E-3]),jnp.zeros(2)))
    gme = JaxGME.GuidedModeExp(phc,gmax=4.0000000001)
    gme.run(kpoints=path,numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)

    alpha = backCost.cost(gme,phc,20)
    ng = (1/(2*jnp.pi))*jnp.abs((gme.kpoints[0,1]-gme.kpoints[0,0])/(gme.freqs[1,20]-gme.freqs[0,20]))

    return([alpha,gme.freqs[0,20],ng])

def varGmax(gmax):
    phc = W1()
    path = jnp.vstack((jnp.array([2.308,2.308+1e-5]),jnp.zeros(2)))
    gme = JaxGME.GuidedModeExp(phc,gmax=gmax)
    gme.run(kpoints=path,numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs)

    alpha = backCost.cost(gme,phc,20)
    ng = (1/(2*jnp.pi))*jnp.abs((gme.kpoints[0,1]-gme.kpoints[0,0])/(gme.freqs[1,20]-gme.freqs[0,20]))

    return([alpha,gme.freqs[0,20],ng])

def varPhiDiv(phidiv):
    phc = W1()
    path = jnp.vstack((jnp.array([2.308]),jnp.zeros(1)))
    gme = JaxGME.GuidedModeExp(phc,gmax=4.000001)
    gme.run(kpoints=path,numeig=21,compute_im=False,verbose=False)

    phc_def_inputs = {}
    backCost = JaxGME.Backscatter(W1,phc_def_inputs,phidiv = phidiv)

    alpha = backCost.cost(gme,phc,20)

    return(alpha)

def normalize(data):
    return data / data[-1]

#%%
rs = jnp.linspace(.1,.45,100)
out = []
t1 = time.time()
for i,r in enumerate(rs):
    if i%10==0:
        print(i)
    out.append(varRad(r))
print(time.time()-t1)
# %%
plt.plot(rs,(jnp.array(out)[:,0]),label=r'$\alpha/n_g^2$')

plt.plot(rs[28],out[28][0],'o',label=f'radius = {round(rs[28],3)}')
plt.yscale('log')
plt.xlabel('radius [a]')
plt.legend()
plt.ylabel(r'$\alpha/n_g^2$')

ax = plt.gca()  # Get the current axis
ax2 = ax.twinx()  # Create a second y-axis
ax2.plot(rs,jnp.array(out)[:,1],'--r',label=r'$n_g$')
ax2.set_ylabel(r'$n_g$')
ax2.legend()
plt.show()

# %%
ks = jnp.linspace(jnp.pi*.5,jnp.pi,10)
vr = []
vr2 = []
t1 = time.time()
for i,k in enumerate(ks):
    if i%10==0:
        print(i)
    vr.append(varK(k,10))
    vr2.append(varK(k,1))
print(time.time()-t1)
# %%
vr = jnp.array(vr)
vr2 = jnp.array(vr2)
plt.plot(vr[:,1],1/(vr[:,0]*vr[:,2]**2))
plt.plot(vr2[:,1],1/(vr2[:,0]*vr2[:,2]**2))
plt.yscale('log')
plt.xlim(.24,.3)
plt.ylim(1,1E7)
plt.show()

# %%

Nys = jnp.arange(30)+5
vny = []
ts = []
for i,Ny in enumerate(Nys):
    t1 = time.time()    
    vny.append(varNy(Ny))
    ts.append(time.time()-t1)
# %%
vny = jnp.array(vny)
vnyNorm = jnp.abs(jnp.vstack((normalize(vny[:,0]),normalize(vny[:,1]),normalize(vny[:,2]))).T)
plt.plot(Nys,(vnyNorm[:,0]*vnyNorm[:,2]**2),label=r'$\alpha$')
plt.plot(Nys,vnyNorm[:,0],label=r'$\alpha/n_g^2$')
plt.plot(Nys,vnyNorm[:,1],label=r'$\omega$')
plt.plot(Nys,vnyNorm[:,2],label=r'$n_g$')
plt.xlabel('Number of holes on either side')
plt.ylabel('Precent of final val')

ax = plt.gca()  # Get the current axis
ax2 = ax.twinx()  # Create a second y-axis
ax2.plot(Nys,ts,'--r',label='compute time')
ax2.set_ylabel('Time [s]')

ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()
#%%
gmaxs = jnp.linspace(2,8,10)
vg = []
ts = []
for i,g in enumerate(gmaxs):
    t1 = time.time()
    vg.append(varGmax(g))
    ts.append(time.time()-t1)
# %%

vg = jnp.array(vg)
vgNorm = jnp.abs(jnp.vstack((normalize(vg[:,0]),normalize(vg[:,1]),normalize(vg[:,2]))).T)
plt.plot(gmaxs,(vgNorm[:,0]*vgNorm[:,2]**2),label=r'$\alpha$')
plt.plot(gmaxs,vgNorm[:,0],label=r'$\alpha/n_g^2$')
plt.plot(gmaxs,vgNorm[:,1],label=r'$\omega$')
plt.plot(gmaxs,vgNorm[:,2],label=r'$n_g$')
plt.xlabel('GMax Values')
plt.ylabel('Precent of final val')

ax = plt.gca()  # Get the current axis
ax2 = ax.twinx()  # Create a second y-axis
ax2.plot(gmaxs,ts,'--r',label='compute time')
ax2.set_ylabel('Time [s]')
ax2.set_yscale('log')

ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

#%%

phidivs = jnp.arange(20)*2+10
vpd = []
ts = []
for i,pd in enumerate(phidivs):
    t1 = time.time()
    vpd.append(varPhiDiv(pd))
    ts.append(time.time()-t1)
# %%

vpd = jnp.array(vpd)
vpdNorm = jnp.abs(normalize(vpd))
plt.plot(phidivs,vpdNorm,label=r'$\alpha/n_g^2$')
plt.xlabel('Divisions of the intigrtal around circle edge')
plt.ylabel('Precent of final val')

ax = plt.gca()  # Get the current axis
ax2 = ax.twinx()  # Create a second y-axis
ax2.plot(phidivs,ts,'--r',label='compute time')
ax2.set_ylabel('Time [s]')

ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()
# %%
rs = jnp.linspace(.15,.35,50)
out = []
t1 = time.time()
for i,r in enumerate(rs):
    print(i)
    out.append(varRadAndK(r))
print(time.time()-t1)
# %%
heat = jnp.array(out)
# %%
alphas = heat[:,0,:]
# %%
plt.imshow(alphas,origin='lower',extent=(.15,.35,.25,.5),aspect='equal',cmap='terrain')
plt.colorbar(label=r'$\alpha/n_g^2$')
r = jnp.linspace(.15,.35,50)[10]
k = jnp.linspace(.25,.5,51)[5]
plt.scatter(r,k,color='red',s=100,label=f"({round(r,2)},{round(k,2)})")
plt.xlim(.15,.35)
plt.ylim(.25,.5)
plt.xlabel('radius [a]')
plt.ylabel(r'k [$2\pi/a$]')
plt.title(f"Min of {round(alphas[5,10],10)}")
plt.legend()
plt.show()
# %%
ngs = heat[:,1,:]
plt.imshow(ngs,origin='lower',extent=(.15,.35,.25,.5),aspect='equal',cmap='terrain',vmax=100,vmin=0)
plt.colorbar(label=r'$n_g$')
plt.xlabel('radius [a]')
plt.ylabel(r'k [$2\pi/a$]')
plt.show()
# %%
from matplotlib.colors import LogNorm
alphaTrue = alphas*ngs**2
plt.imshow(alphaTrue,origin='lower',extent=(.15,.35,.25,.5),aspect='equal',cmap='terrain',norm=LogNorm(vmin=1E-7, vmax=1E-1))
plt.colorbar(label=r'$\alpha$')
r = jnp.linspace(.15,.35,50)[10]
k = jnp.linspace(.25,.5,51)[5]
plt.scatter(r,k,color='red',s=100,label=f"({round(r,2)},{round(k,2)})")
plt.xlim(.15,.35)
plt.ylim(.25,.5)
plt.xlabel('radius [a]')
plt.ylabel(r'k [$2\pi/a$]')
plt.title(f"Min of {round(alphaTrue[5,10],10)}, ng of {round(ngs[5,10],2)}")
plt.legend()
plt.show()

