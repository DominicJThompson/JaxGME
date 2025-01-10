import jax
import jax.numpy as jnp

def brent_method(f, a, b, tol=1e-5, max_iter=100):
    """
    Implementation of Brent's Method for root-finding in JAX.
    
    Args:
        f: The function for which to find the root.
        a: The lower bound of the interval.
        b: The upper bound of the interval.
        tol: Tolerance for stopping criterion.
        max_iter: Maximum number of iterations.
    
    Returns:
        A root of the function within the interval [a, b].
    """
    # Ensure f(a) and f(b) have opposite signs
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    # Initialize variables
    c = a
    fc = fa
    d = e = b - a
    for iteration in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        
        if jnp.abs(fc) < jnp.abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        
        tol_act = 2 * jnp.finfo(float).eps * jnp.abs(b) + tol / 2
        m = 0.5 * (c - b)

        if jnp.abs(m) <= tol_act or fb == 0.0:
            return b  # Root found
        
        if jnp.abs(e) >= tol_act and jnp.abs(fa) > jnp.abs(fb):
            s = fb / fa
            if a == c:  # Secant method
                p, q = 2 * m * s, 1 - s
            else:  # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            
            if p > 0:
                q = -q
            p = jnp.abs(p)
            
            if 2 * p < jnp.minimum(3 * m * q - jnp.abs(tol_act * q), jnp.abs(e * q)):
                e, d = d, p / q
            else:
                d = m
                e = d
        else:
            d = m
            e = d

        a, fa = b, fb
        if jnp.abs(d) > tol_act:
            b += d
        else:
            b += jnp.sign(m) * tol_act
        fb = f(b)

    raise RuntimeError("Maximum number of iterations reached.")


#This function has been optomized for finding the frequencies
#of the slab guided modes
def fsolve_jax(f, lb, ub, g, eps_array, d_array):
    """
    Solve for scalar f(x, *args) = 0 w.r.t. scalar x within lb < x < ub using self implemented brent method

    g, eps_array, and d_array specify the function to being used for slab guided modes
    """
    def f_opt(x):
        return(f(x,g,eps_array,d_array)[0])
    
    return brent_method(f_opt, lb, ub)


def toeplitz_jax(c, r=None):
    """
    Constructs a Toeplitz matrix using JAX.
    Args:
        c (jax.numpy.ndarray): The first column of the Toeplitz matrix.
        r (jax.numpy.ndarray, optional): The first row of the Toeplitz matrix.
            Defaults to `None`, in which case `r = conj(c)`.

    Returns:
        jax.numpy.ndarray: The Toeplitz matrix.
    """
    if r is None:
        r = jnp.conj(c)
    else:
        r = r.at[0].set(c[0])  # Ensure consistency between first column and first row

    n = len(c)
    m = len(r)

    # Create an index matrix for broadcasting
    row_idx = jnp.arange(n)[:, None]
    col_idx = jnp.arange(m)
    idx = row_idx - col_idx  # Difference between row and column indices

    # Map index to the appropriate values
    return jnp.where(idx >= 0, c[idx], r[-idx])


def toeplitz_block_jax(n, T1, T2):
    """
    Constructs a Hermitian Toeplitz-block-Toeplitz matrix with n blocks and 
    T1 in the first row and T2 in the first column of every block in the first
    row of blocks.

    Args:
        n (int): Number of blocks along one dimension.
        T1 (jax.numpy.ndarray): The first row of the blocks.
        T2 (jax.numpy.ndarray): The first column of the blocks.

    Returns:
        jax.numpy.ndarray: A Hermitian Toeplitz-block-Toeplitz matrix.
    """
    ntot = T1.shape[0]
    p = ntot // n  # Linear size of each block
    Tmat = jnp.zeros((ntot, ntot), dtype=T1.dtype)

    for ind1 in range(n):
        for ind2 in range(ind1, n):
            toep1 = T1[(ind2 - ind1) * p:(ind2 - ind1 + 1) * p]
            toep2 = T2[(ind2 - ind1) * p:(ind2 - ind1 + 1) * p]
            block = toeplitz_jax(toep2, toep1)
            Tmat = Tmat.at[ind1*p:(ind1+1)*p, ind2*p:(ind2+1)*p].set(block)

    # Make the matrix Hermitian
    Tmat = jnp.triu(Tmat) + jnp.conj(jnp.transpose(jnp.triu(Tmat, 1)))

    return Tmat

def extend_jax(vals, inds, shape):
    """
    Creates an array of shape `shape` where indices `inds` have values `vals`.
    
    Parameters:
    vals (jax.numpy.ndarray): Values to insert.
    inds (tuple of arrays): Tuple containing the indices where values will be inserted.
    shape (tuple): Shape of the resulting array.
    
    Returns:
    jax.numpy.ndarray: Array of the given shape with values inserted at specified indices.
    """
    z = jnp.zeros(shape, dtype=vals.dtype)  # Create an array of zeros with the desired shape and dtype
    z = z.at[inds].set(vals)  # Use JAX's indexed update to insert values at specific indices
    return z

def spdot_jax(sparse_matrix, jax_array):
    """
    Perform a differentiable dot product between a scipy sparse matrix and a jax array.

    Args:
        sparse_matrix (scipy.sparse.spmatrix): A sparse matrix in scipy format.
        jax_array (jax.numpy.ndarray): A dense array compatible with JAX.

    Returns:
        jax.numpy.ndarray: The resulting dense array from the sparse dot product.
    """
    # Ensure the sparse matrix is in COO format
    coo = sparse_matrix.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data

    # Perform the sparse-dense multiplication
    result = jnp.zeros((coo.shape[0],), dtype=jax_array.dtype)
    for r, c, d in zip(rows, cols, data):
        result = result.at[r].add(d * jax_array[c])
    
    return result

@jax.jit
def eigh_jit_jax(matrix):
    """
    Compute the eigenvalues and eigenvectors of a Hermitian (symmetric) matrix using jnp.linalg.eigh,
    wrapped with JAX's jit for improved performance.

    Parameters:
        matrix (jnp.ndarray): A Hermitian matrix (symmetric if real).

    Returns:
        tuple: A tuple (eigenvalues, eigenvectors) where:
            - eigenvalues is a 1D array containing the eigenvalues in ascending order.
            - eigenvectors is a 2D array where each column is an eigenvector.
    """
    return jnp.linalg.eigh(matrix)

def find_nearest_jax(array, value, N):
    """
    Find the indexes of the N elements in an array nearest to a given value
    (Not the most efficient way but this is not a coding interview...) (its ok, not a large bottle neck)
    Implementation with jax
    """
    idx = jnp.abs(array - value).argsort()
    return idx[:N]