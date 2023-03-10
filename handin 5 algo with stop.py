import numpy as np
from case_studies import *
from scipy.optimize import *

def back_track(f,df,x,p,a=1,c1=0.01,rho=0.5):
    while f(np.array(x+a*p)) > f(np.array(x)) + c1*a* p.T@df(np.array(x)): #Check if a fulfills sufficient decrease conditions
        a = a*rho #If not, we decrease a by a factor rho
    return(a)


def constrained_newton(x0, A, f, df, hf, iterations=1000):
    """
    Estimates the local minimum of f using the Newton algorithm with equality constarints
    """
    x_k = x0
    for i in range(iterations):

        # Check if hessian is positive definit
        if np.all(np.linalg.eigvals(hf(x_k)) > 0):
            B = hf(x_k)

        else:
            # compute approx hessian
            values, vectors = np.linalg.eig(hf(x_k))
            for i in range(len(values)):
                B = + np.abs(values[i]) * np.outer(vectors[i], vectors[i])

        # create the linear system
        matrix_A = np.block([[B, A.T], [A, np.zeros([A.shape[0], A.shape[0]])]])

        matrix_B = np.block([-df(x_k), np.zeros(A.shape[0])])

        # solve the linear system and extract pk
        solved = np.linalg.solve(matrix_A, matrix_B)

        p_k = solved[0:A.shape[1]]

        lam = solved[A.shape[1]:]
        #print('stop', np.linalg.norm(df(x_k) + A.T @ lam))

        # Compute the step size
        alpha_k = back_track(x=x_k, p=p_k, f=f, df=df)

        # Update the current solution
        x_k = x_k + alpha_k * p_k
        if np.linalg.norm(df(x_k)+A.T@lam) < 10**(-6): #stopping criteria
            break
    return -x_k, i + 1

m = 2
n = 3

A = np.random.rand(m,n)
x = np.random.rand(n)
b = np.random.rand(1)

x0 = x - np.linalg.pinv(A) @ (A@x + b)

x_newton,iterations_newton = constrained_newton(x0=np.array(x0),A=A,f=f5,df=df5,hf=Hf5)
print(iterations_newton)

def constrained_steepest_descent(f, df, x, A, max_iterations=10000, rho=0.5):
    """
    f is function that is being optimized
    df is the gradient of f
    x is staring point
    """
    beta = 1
    M = np.eye(A.shape[1]) - A.T @ (np.linalg.inv(A @ A.T)) @ A
    for i in range(0, max_iterations):
        pk = -M @ df(x)  # we set the ray direction

        ak = back_track(f=f, df=df, x=x, p=pk, a=beta)  # We find the step size

        x = x + ak * pk  # we update x with the stepsize and direction

        #lam = np.linalg.norm(np.linalg.inv(A @ A.T) @ A @ df(x))  # we calculate lambda with the eq we found in theory

        # print("stop",np.linalg.norm(df(x)+np.inner(A,lam)))
        if np.linalg.norm(M.dot(df(x))) < 10 ** (-5):
            break

        beta = ak / rho  # We calculate the new initial step size

    return (-x, i + 1)


m = 1
n = 3

A = np.random.rand(m,n)
x = np.random.rand(n)
b = np.random.rand(1)

x0 = x - np.linalg.pinv(A) @ (A@x + b)

x,iterations = constrained_steepest_descent(f=f3,df=df3,x=np.array(x0),A=A)


m = 19
n = 20

A = np.random.rand(m,n)
x = np.random.rand(n)
b = np.random.rand(1)

x0 = x - np.linalg.pinv(A) @ (A@x + b)

x_newton,iterations_newton = constrained_newton(x0=np.array(x0),A=A,f=f3,df=df3,hf=Hf3)
x_sd,iterations_sd = constrained_steepest_descent(f=f3,df=df3,x=np.array(x0),A=A)
print(iterations_newton)
print("Newton's algorithm:\n iterations:",iterations_newton,"point found:",x_newton)

print("Steepest descent algorithm:\n iterations:",iterations_sd,"point found:",x_sd)


linear_constraints = {"type": "eq", "fun": lambda x: A.dot(x) - b}

opt = minimize(f3,x0=x0,constraints=linear_constraints)
print(opt.x)