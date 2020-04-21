from __future__ import division
import numpy as np
from scipy.integrate import simps
from scipy.sparse import spdiags, coo_matrix, csc_matrix, dia_matrix, dok_matrix, identity
from scipy.sparse.linalg import spsolve
from itertools import chain
import pylab

import scipy
print("scipy", scipy.__version__)
print("numpy", np.__version__)

# Solves the nonlinear Fisher's equation implicit/explict discretisation 
# (Crank-Nicolson method) for the diffusion term and explicit for the
# reaction term using time stepping. Dirichlet BCs on l.h.s and r.h.s..
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


def M(x):
    """Returns coefficient matrix for diffusion equation (constant coefficients)"""
    ones = np.ones(len(x))
    M =  spdiags( [ones, -2*ones, ones], (-1,0,1), len(x), len(x) ).tocsc()
    J = len(x)
    M[J-1,J-2] = 2.0 # Neumann conditions
    return M
    
def set_DBCs(M):
    M = M.todok()
    M[0,0] = 1;     M[0,1] = 0;
    M[1,0] = 0;
    M[-2,-1] = 0
    M[-1,-1] = 1;   M[-1,-2] = 0
    return M.tocsc()


def bc_vector(x, d, h, left_value, right_value):
    bc = dok_matrix((len(x), 1)) # column vector
    bc[0,0]  = -left_value
    bc[1,0]  = d/h**2 * left_value
    bc[-2,0] = d/h**2 * right_value
    bc[-1,0] = -right_value
    return bc.tocsc()

def Imod(x):
    data = np.ones(len(x))
    data[0] = 0
    data[-1] = 0
    offset = 0
    shape = (len(x),len(x))
    Imod = dia_matrix((data, offset), shape=shape).todok()
    return Imod.tocsc()

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
h = x_range / len(x)
d = 0.1
B = 1.0
s = d / h**2
tau = 0.5
fsn = 1

# Initial conditions (needs to end up as a column vector)
u_init = np.matrix( solution_fishers_eqn(x,0.0,d,B,fsn) ).reshape((len(x),1))
u = u_init


# Matrices and vectors
M = M(x)                # Coefficient matrix
Im = Imod(x)            # Modifed identity matrix
I = identity(len(x))    # Standard identity matrix
theta = 0.5 * np.ones(len(x))   # IMEX but with a fully
theta[0]  = 0.0                   # implicit at boundary
theta[-1] = 0.0                   # points.
theta_matrix = dia_matrix((theta, 0), shape=(len(x),len(x))).tocsc()    
theta_matrix_1m = dia_matrix((1-theta, 0), shape=(len(x),len(x))).tocsc()

# Boundary conditions
#M = set_DBCs(M)
left_value = 1.0
right_value = 0.0
bc = bc_vector(x, d, h, left_value, right_value)
print ("left_value", left_value)
print ("right_value", right_value)

# Cache
slices = [x]

# Time stepping loop
STEPS = 20
PLOTS = 5
for i in range(0, STEPS+1):
    
    u_array = np.asarray(u).flatten()
    # Reaction (nonlinear, calculated everytime)
    r = np.matrix(B*u_array**(fsn)*(1 - u_array)).reshape((len(x),1))
    
    # Export data
    
    if i % abs(STEPS/PLOTS) == 0:
        
        plot_num = abs(i/(STEPS/PLOTS))
        fraction_complete = plot_num / PLOTS
        print ("fraction complete", fraction_complete)
        
        print ("I: %g" % (simps(u_array,dx=h), ) )
        pylab.figure(1)
        pylab.plot(x,u_array, "o", color=pylab.cm.Accent(fraction_complete))
        pylab.plot(x, solution_fishers_eqn(x, i*tau, d, B, fsn), "-", color=pylab.cm.Accent(fraction_complete))
        slices.append(u_array)
    
    # Prepare l.h.s. @ t_{n+1}
    A = (I - tau*s*theta_matrix*M)
    
    # Prepare r.h.s. @ t_{n}
    b = csc_matrix( (I + tau*s*theta_matrix_1m*M)*u + tau*r)
    
    # Set boundary conditions
    b[0,0] = left_value
    #b[-1,-1] = right_value
    
    # Solve linear 
    u = spsolve(A,b)                       # Returns an numpy array,
    u = np.matrix(u).reshape((len(x),1))   # need a column matrix
    

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

np.savetxt("slices.txt", np.column_stack(slices))
pylab.title("Time-step method (dots), analytical results (lines) with dt=%g." % tau)
pylab.xlabel("x")
pylab.ylabel("u(x)")
pylab.ylim(ymax=1.2)
pylab.savefig("timestep_solution.pdf")
pylab.show()


