from __future__ import division
import scipy
import numpy
print "scipy", scipy.__version__
print "numpy", numpy.__version__

import numpy as np
from scipy.sparse.linalg import inv
from scipy.integrate import simps
from scipy.sparse import spdiags, coo_matrix, csc_matrix, dia_matrix, dok_matrix, identity
from scipy.sparse.linalg import spsolve
from itertools import chain
import pylab


# Solves the nonlinear Fisher's equation implicit/explict discretisation 
# (Crank-Nicolson method). A Newton iteration is used to find the unknown
# solution variable at the future time step. Dirichlet BCs on l.h.s and
# r.h.s.
# 
# u_t = d u_{xx} + B u (1 - u)
#
# BUGS: Neumann boundary conditions on the r.h.s. do not work correctly
#       but provided we don't iterate the equation to forward in time
#       the analytical solution can be retrieved (because the boundary
#       point does not play a role in the solution).
#
#       The analytical experssion is only valid for DBCs on the l.h.s.
#       and NBCs on the r.h.s., but again let's not time step too far
#       so that we get good agreement.


def J(x,u,a,b,h,theta,n=1):
    """The Jacobian of Fishers equation with respect to the finite difference stencil."""
    ones = np.ones(len(u))
    u  = np.asarray(u).flatten()
    dFdu1j = a / h**2 * (1-theta) * np.ones(len(u))
    if n == 1:
        dFuj = -2*a/h**2 * (1-theta) * ones - b * u * (1 - theta) + b * (1 - theta) * (1 - u)
    elif n == 2:
        dFuj = -2*a/h**2 * (1-theta) - b * u * (1-theta) + b * (1-theta) * (1 - u)
    else:
        raise ValueError("The `n` kwargs must be an integer with the value 1 or 2.")
    dFduj1 = a / h**2 * (1-theta) * np.ones(len(u))
    return spdiags( [dFdu1j, dFuj, dFduj1], (-1,0,1), len(x), len(x) ).tocsc()
    
def M(x):
    """Returns coefficient matrix for diffusion equation (constant coefficients)
    with Dirichlet BC on the left and Neumann BC on the right."""
    ones = np.ones(len(x))
    M = spdiags( [ones, -2*ones, ones], (-1,0,1), len(x), len(x) ).todok()
    J = len(ones)
    M[J-1,J-2] = 2.0 # Neumann conditions
    return M.tocsc()

def solution_fishers_eqn(x,t,d,B,n):
    """Analytical solution to Fishers equation with u(x0,t)=1
    and u_x(xJ,t)=0. The parameter n can be either 1 or 2."""
    if n == 1:
        U = 5 * np.sqrt(d * B / 6)
        return 1.0 / (1 + np.exp(np.sqrt(B/(6*d))*(x - U*t)) )**2
    elif n == 2:
        V = np.sqrt(d*B/2)
        return 1.0/ (1 + np.exp(np.sqrt(B/(2*d))*(x - V*t)) )
    else:
        raise ValueError("The parameter `n` must be an integer with the value 1 or 2.")


# Coefficients and simulation constants
x = np.linspace(-10, 10, 100)
x_range = np.max(x) - np.min(x)
h = x_range / len(x)    # mesh spacing
d = 0.1                 # diffusion coefficient
B = 1.0                 # reaction constant
s = d / h**2
tau = 0.5               # time step
fsn = 1                 # fishers 'n' constant (either 1 or 2)

# Boundary conditions
left_value = 1.0
right_flux = 0.0

# Initial conditions (needs to end up as a column vector)
u_init = np.matrix( solution_fishers_eqn(x,0.0,d,B,fsn) ).reshape((len(x),1))
u_init[0] = left_value # This initial value is how DBCs are incorporated the lhs is also the lhs DBC
u = u_init

# Matrices and vectors
M = M(x)                # Coefficient matrix
I = identity(len(x))    # Standard identity matrix
theta = 0.5

# Cache
slices = [x]

# Time stepping loop
STEPS = 20
PLOTS = 5
for i in range(0, STEPS+1):
    
    u_array = np.asarray(u).flatten()
    
    if i % abs(STEPS/PLOTS) == 0:
        plot_num = abs(i/(STEPS/PLOTS))
        fraction_complete = plot_num / PLOTS
        print "fraction complete", fraction_complete
        print "I: %g" % (simps(u_array,dx=h), )
        pylab.figure(1)
        pylab.plot(x,u_array, "o", color=pylab.cm.Accent(fraction_complete))
        pylab.plot(x, solution_fishers_eqn(x, i*tau, d, B, fsn), "-", color=pylab.cm.Accent(fraction_complete))
        slices.append(u_array)
        
    # Reaction (nonlinear, calculated everytime)
    r = np.matrix(B*u_array**(fsn)*(1 - u_array)).reshape((len(x),1))
    
    # Prepare Newton-Raphson variables
    un = np.matrix(u_array).transpose()
    r = np.matrix(B*np.asarray(u)**(fsn)*(1 - np.asarray(u)))
    Fn = s*M*u + r
    Fn[0] = 0.0     # DBC on the l.h.s.
    Fn[-1] = 0.0    # DBC on r.h.s
    vk = np.matrix(un + tau*Fn) # Guess value
    
    # Modified newton iteration pg. 127 Hundsdorfer
    for k in range(0,10):
        
        r = B*np.asarray(vk)**(fsn)*(1 - np.asarray(vk))
        Fk = s*M*vk + r
        Fk[0] = 0.0 # DBC on the l.h.s.
        Fk[-1] = 0.0 # DBC on r.h.s.
        g = vk - un - (1-theta)*tau*Fn - theta*tau*Fk
        
        J_term = I - theta*tau*J(x,un,d,B,h,theta,n=fsn)
        J_term_inv = inv(J_term)
        vk1 = vk - J_term_inv*g
        vk = vk1
    
    u = vk
    
    

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

np.savetxt("slices.txt", np.column_stack(slices))
pylab.title("Modified Newton solver (dots), analytical results (lines) with dt=%g." % tau)
pylab.xlabel("x")
pylab.ylabel("u(x)")
pylab.ylim(ymax=1.2)
pylab.savefig("newton_solution.pdf")
pylab.show()


