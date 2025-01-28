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
manager = ConstraintManager(x0=vars,numberHoles=3)

#manager.add_inside_unit_cell('Inside')
#manager.add_min_rad('minimumRadius',.1)
manager.add_min_dist('name',.1,3,W1Vars(NyChange=3+3))
# %%
manager.constraintsDisc
#%%