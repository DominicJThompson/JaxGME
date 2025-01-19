"""
Backend for the simulations. Available backends:
 - numpy [default]
 - jax
A backend can be set with the 'set_backend'
    import legume
    legume.set_backend("jax")

Everything is done in either numpy or jax depending on backend.
"""

# Numpy must be present
import numpy as np
import scipy as sp
from scipy import sparse
# Import some specially written functions
from .utils import toeplitz_block, fsolve, extend, find_nearest


#import jax if available
try:
    import jax.numpy as jnp
    import jax.scipy as jsp
    #import primitives 
    from .primitives_Jax import (fsolve_jax,toeplitz_block_jax,extend_jax,spdot_jax,
                                 eigh_jit_jax, find_nearest_jax)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class Backend(object):
    """
    Backend Base Class 
    """
    # types
    int = np.int64
    float = np.float64
    complex = np.complex128

    #neccessary since jax needs alternatives
    ndarray = np.ndarray
    argset = staticmethod(lambda arr, pos, vals: (arr.__setitem__(pos, vals) or arr))
    int_ = staticmethod(np.int_)
    pi = np.pi
    tile = staticmethod(np.tile)
    argmin = staticmethod(np.argmin)
    remainder = staticmethod(np.remainder)
    append = staticmethod(np.append)
    inner = staticmethod(np.inner)

    #functions i have added to the code
    diff = staticmethod(np.diff)
    sign = staticmethod(np.sign)
    find_nearest = staticmethod(find_nearest)

    def __repr__(self):
        return self.__class__.__name__


class NumpyBackend(Backend):
    """ Numpy Backend """

    # methods
    sum = staticmethod(np.sum)
    stack = staticmethod(np.stack)
    hstack = staticmethod(np.hstack)
    vstack = staticmethod(np.vstack)
    transpose = staticmethod(np.transpose)
    reshape = staticmethod(np.reshape)
    toeplitz_block = staticmethod(toeplitz_block)
    roll = staticmethod(np.roll)
    where = staticmethod(np.where)
    argwhere = staticmethod(np.argwhere)
    triu = staticmethod(np.triu)
    amax = staticmethod(np.amax)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    sort = staticmethod(np.sort)
    argsort = staticmethod(np.argsort)
    interp = staticmethod(np.interp)
    fsolve_D22 = staticmethod(fsolve)
    extend = staticmethod(extend)
    round = staticmethod(np.round)
    shape = staticmethod(np.shape)
    concatenate = staticmethod(np.concatenate)
    size = staticmethod(np.size)
    full = staticmethod(np.full)
    unique = staticmethod(np.unique)
    meshgrid = staticmethod(np.meshgrid)

    # math functions
    exp = staticmethod(np.exp)
    bessel1 = staticmethod(sp.special.j1)
    sqrt = staticmethod(np.sqrt)
    divide = staticmethod(np.divide)
    abs = staticmethod(np.abs)
    square = staticmethod(np.square)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    tanh = staticmethod(np.tanh)
    norm = staticmethod(np.linalg.norm)
    dot = staticmethod(np.dot)
    cross = staticmethod(np.cross)
    real = staticmethod(np.real)
    imag = staticmethod(np.imag)
    inv = staticmethod(np.linalg.inv)
    eig = staticmethod(np.linalg.eig)
    eigh = staticmethod(np.linalg.eigh)
    eigsh = staticmethod(sp.sparse.linalg.eigsh)
    outer = staticmethod(np.outer)
    conj = staticmethod(np.conj)
    var = staticmethod(np.var)
    power = staticmethod(np.power)
    matmul = staticmethod(np.matmul)
    tan = staticmethod(np.tan)
    arctan = staticmethod(np.arctan)
    log10 = staticmethod(np.log10)

    # Dot product between a scipy sparse matrix and a numpy array.
    spdot = staticmethod(lambda spmat, mat: spmat.dot(mat))

    # Sparse matrix class
    #coo = staticmethod(sparse.coo_array)

    def is_array(self, arr):
        """ check if an object is an array """
        return isinstance(arr, np.ndarray)

    # constructors
    diag = staticmethod(np.diag)
    array = staticmethod(np.array)
    ones = staticmethod(np.ones)
    zeros = staticmethod(np.zeros)
    eye = staticmethod(np.eye)
    linspace = staticmethod(np.linspace)
    arange = staticmethod(np.arange)
    newaxis = staticmethod(np.newaxis)
    zeros_like = staticmethod(np.zeros_like)


if JAX_AVAILABLE:

    class JaxBackend(Backend):
        """ Jax Backend """

        # types
        int = jnp.int64
        float = jnp.float64
        complex = jnp.complex128

        #things needed for jax addtion
        ndarray = jnp.ndarray
        argset = staticmethod(lambda arr, pos, vals: arr.at[pos].set(vals))
        int_ = staticmethod(jnp.int_)
        pi = jnp.pi
        tile = staticmethod(jnp.tile)
        argmin = staticmethod(jnp.argmin)
        remainder = staticmethod(jnp.remainder)
        append = staticmethod(jnp.append)
        inner = staticmethod(jnp.inner)

        #functions i have added to the code
        diff = staticmethod(jnp.diff)
        sign = staticmethod(jnp.sign)
        find_nearest = staticmethod(find_nearest_jax)

        # methods
        sum = staticmethod(jnp.sum)
        stack = staticmethod(jnp.stack)
        hstack = staticmethod(jnp.hstack)
        vstack = staticmethod(jnp.vstack)
        transpose = staticmethod(jnp.transpose)
        reshape = staticmethod(jnp.reshape)
        toeplitz_block = staticmethod(toeplitz_block_jax)
        roll = staticmethod(jnp.roll)
        where = staticmethod(jnp.where)
        argwhere = staticmethod(jnp.argwhere)
        triu = staticmethod(jnp.triu)
        amax = staticmethod(jnp.amax)
        max = staticmethod(jnp.max)
        min = staticmethod(jnp.min)
        sort = staticmethod(jnp.sort)
        argsort = staticmethod(jnp.argsort)
        interp = staticmethod(jnp.interp)
        fsolve_D22 = staticmethod(fsolve_jax)
        extend = staticmethod(extend_jax)
        round = staticmethod(jnp.round)
        shape = staticmethod(jnp.shape)
        concatenate = staticmethod(jnp.concatenate)
        size = staticmethod(jnp.size)
        full = staticmethod(jnp.full)
        unique = staticmethod(jnp.unique)
        meshgrid = staticmethod(jnp.meshgrid)

        # math functions
        exp = staticmethod(jnp.exp)
        sqrt = staticmethod(jnp.sqrt)
        divide = staticmethod(jnp.divide)
        abs = staticmethod(jnp.abs)
        square = staticmethod(jnp.square)
        sin = staticmethod(jnp.sin)
        cos = staticmethod(jnp.cos)
        tanh = staticmethod(jnp.tanh)
        cross = staticmethod(jnp.cross)
        norm = staticmethod(jnp.linalg.norm)
        dot = staticmethod(jnp.dot)
        real = staticmethod(jnp.real)
        imag = staticmethod(jnp.imag)
        inv = staticmethod(jnp.linalg.inv)
        eig = staticmethod(jnp.linalg.eig)
        eigh = staticmethod(eigh_jit_jax)
        #eigsh = staticmethod(jsp.sparse.linalg.eigsh)
        outer = staticmethod(jnp.outer)
        conj = staticmethod(jnp.conj)
        var = staticmethod(jnp.var)
        power = staticmethod(jnp.power)
        matmul = staticmethod(jnp.matmul)
        tan = staticmethod(jnp.tan)
        arctan = staticmethod(jnp.arctan)
        log10 = staticmethod(jnp.log10)

        #this function has convergance issues, need to check input values are within the 
        #convergent region. Can change n_iter to alter region
        bessel1 = staticmethod(lambda array: jsp.special.bessel_jn(array,v=1,n_iter=30)[1])

        # Dot product between a scipy sparse matrix and a numpy array.
        # Differentiable w.r.t. the numpy array.
        spdot = staticmethod(spdot_jax)

        # constructors
        diag = staticmethod(jnp.diag)
        array = staticmethod(jnp.array)
        ones = staticmethod(jnp.ones)
        zeros = staticmethod(jnp.zeros)
        eye = staticmethod(jnp.eye)
        linspace = staticmethod(jnp.linspace)
        arange = staticmethod(jnp.arange)
        newaxis = staticmethod(jnp.newaxis)
        zeros_like = staticmethod(jnp.zeros_like)


backend = NumpyBackend()


def set_backend(name):
    """
    Set the backend for the simulations.
    This function monkey-patches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.
    
    Parameters
    ----------
    name : {'numpy', 'autograd'}
        Name of the backend. HIPS/autograd must be installed to use 'autograd'.
    """
    # perform checks
    if name == 'autograd' and not AG_AVAILABLE:
        raise ValueError("Autograd backend is not available, autograd must \
            be installed.")
    
    # perform checks
    if name == 'jax' and not JAX_AVAILABLE:
        raise ValueError("Jax backend is not available, jax must \
            be installed.")

    # change backend by monkeypatching
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'autograd':
        backend.__class__ = AutogradBackend
    elif name == 'jax':
        backend.__class__ = JaxBackend
    else:
        raise ValueError(f"unknown backend '{name}'")
