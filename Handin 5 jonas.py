import numpy as np
from case_studies import *
from scipy.optimize import *


def back_track(f,df,x,p,a=1,c1=0.01,rho=0.5):

    while f(x+a*p) > f(x) + c1*a* np.dot(p,df(x)): #Check if a fulfills sufficient decrease conditions
        #print('value a ', a)
        #print('while statement:', f(np.array(x+a*p)), f(np.array(x)) + c1 * a * p.transpose()@df(np.array(x)))
        a = a*rho #If not, we decrease a by a factor rho
        #print('something wrong here?')
    return(a)

def constrained_newton(x0, A, f, df, hf, iterations=1000):
    """
    Estimates the local minimum of f using the Newton algorithm with equality constarints
    """
    x_k = x0
    print('x_k ', x_k)
    for i in range(iterations):
        #print(i)

        # Check if hessian is positive definit
        if np.all(np.linalg.eigvals(hf(x_k)) > 0):
            B = hf(x_k)
            # print("hessian",B)
        else:
            # compute approx hessian
            values, vectors = np.linalg.eig(hf(np.array(x_k)))
            for i in range(len(values)):
                B = + np.abs(values[i]) * np.outer(vectors[i], vectors[i])

        # create the linear system
        #matrix_Left = np.block([[B, A.T], [A, np.zeros(A.shape[0])]])
        AT = A.transpose()
        B_AT = np.concatenate((B,AT),axis=1)
        zeroMatrix = np.zeros((A.shape[0],A.shape[0]))

        A_0 = np.concatenate((A,zeroMatrix),axis=1)
        #print('B_AT; ', B_AT)
        #print('A0;', A_0)
        B_AT_above_A_0 = np.concatenate((B_AT,A_0),axis=0)
        #print('full matrix: ', B_AT_above_A_0)
        dfAsArrayT = np.array([-df(x_k)])
        #print('dfassarrayT,', dfAsArrayT)
        dfAsArray = dfAsArrayT.transpose()
        zeroVectorT = np.array([np.zeros((A.shape[0]),)])
        #print('zerovectorT;', zeroVectorT)
        zeroVector = zeroVectorT.transpose()

        matrixRight = np.concatenate((dfAsArray,zeroVector),axis=0)

        inv_leftMatrix = np.linalg.inv(B_AT_above_A_0)
        # solve the linear system and extract pk
        solved = inv_leftMatrix@matrixRight

        #print('solved;', solved)
        #matrix_Right = np.block([-df(x_k), np.zeros(A.shape[0])])
        solved1 = np.linalg.solve(B_AT_above_A_0, matrixRight)
        #print('solved1 ', solved1)
        p_k = solved[:A.shape[1]]
        print("pk",p_k)
        lam = solved[A.shape[1]:]
        #print('lam;', lam)
        print("stop", np.linalg.norm(df(x_k) - AT@lam), np.linalg.norm(df(x_k)))
        #print("pk", p_k)
        #print("gradient", np.linalg.norm(df(x_k)))
        # Compute the step size
        flatP_k = p_k.flatten()

        alpha_k = back_track(x=x_k, p=flatP_k, f=f, df=df)
        #print('alpha', alpha_k)
        # Update the current solution
        x_k = x_k + alpha_k * flatP_k
        # Check stopping condition
        #if np.linalg.norm(df(x_k)) < 1e-6:
         #   break
        if np.linalg.norm(df(x_k) - AT @ lam) < 10 ** (-5):
            break
        # if np.linalg.norm(lam) < stop_eps:
        # break

    return x_k

A = np.array([[1,7,7],[6,4,-4]])
#H = A@np.array([0,1])
#print('H', H)
x = np.array([2,4,8])
b = np.array([-1,1,9])

x0 = np.array(x - np.linalg.pinv(A) @ A@(x + b))
print('A: ', A)
Solution = constrained_newton(x0=np.array(x0),A=A,f=f3,df=df3,hf=Hf3)
print(Solution)
