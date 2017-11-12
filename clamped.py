from numpy import sin, cos, linspace, ones, zeros, arange, exp, sum
from numpy import append, diff
from numpy import concatenate
from scipy.io import savemat
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
#ode integrator
from scipy.integrate import odeint
import time

from multilink import Multilink

class Clamped(Multilink):
    """
    Class for the clamped case

    Attributes
    ----------
    Nlink : int
        Number of links.
    gamma : float
        Resistance ratio from RFT, gamma >1.
    spring_const : float
        Dimensionless spring constant.
    sp : float
        Sperm number.
    epsilon : float
        Dimensionless oscillation amplitude
    actuation : lambda
        actuation formula
    """
    def __init__(self, Nlink, gamma, sp, epsilon, f_y1, f_actuation):
        super(Clamped, self).__init__(Nlink, gamma, sp)
        self.epsilon = epsilon
        self.f_y1 = f_y1
        self.f_actuation = f_actuation
        self.x2 = 2/self.N
        self.y2 = f_y1(0)

    def stiffness_distribution(self, stiffness_type, stiffness_params):
        N = self.N
        self.spring_positions = linspace(0,1-1/N,N)
        if stiffness_type == 'linear':
            a, b = stiffness_params
            C = a*self.spring_positions+b
            self.C_distribution = C[1:]
        elif stiffness_type == 'quadratic':
            a,b,c = stiffness_params
            C = a*self.spring_positions**2 + b*self.spring_positions + c
            self.C_distribution = C[1:]
        elif stiffness_type == 'exponential':
            a = stiffness_params
            C = exp(a*self.spring_positions)
            self.C_distribution = C[1:]
        else:
            raise ValueError(
                'spring distribution should be one of linear, quadratic, or exponential, while the input given is: %s' % stiffness_type)

        if min(self.C_distribution) <= 0:
            raise ValueError(
                'spring distirbution should have positive values, while the minimum value of the stiffness distribution is: %.1f' % min(self.C_distribution)
            )
        return

    def lhs(self, x, y, theta):
        """
        The system is represented by A y' = f(y,t), where
        y = [x1,x2,...xN, y1, y2, ..., yN, theta1, theta2, ..., thetaN].
        3N-5 equations in total.
        We calculate A in this function.
        eq1: torque balance of last N-1 links evaluated at x2
        eq2: torque balance of last N-2 links evaluated at x3
        eq3: torque balance of last N-3 links evaluated at x4
        ...
        ...
        ...
        eq (N-1): torque balance of the last link evaluated at xN
        eq(N): xdot1 = 0
        eq(N+1): xdot2 = 1/N
        eq(N+2): 0  - xdot3 - 1/N*thetadot2*sin(theta2) = 0
        eq(N+3):  xdot3 - xdot4 - 1/N*thetadot3*sin(theta3) = 0
        ...
        ...
        ...
        eq(2N): xdotN - xdot_{N+1} - 1/N*thetadotN*sin(thetaN) = 0
        eq(2N+1): ydot1 = -epsilon*sin(t)
        eq(2N+2): ydot2 = -epsilon*sin(t)
        eq(2N+3): -epsilon*sin(t) - ydot3 + 1/N*thetadot2*cos(theta2) = 0
        eq(2N+4): ydot3 - ydot4 +1/N*thetadot3*cos(theta3) = 0
        ...
        ...
        ...
        eq(3N+1): ydotN - ydot_{N+1} - 1/N*thetadotN*cos(thetaN) = 0

        """
        N = self.N
        A = lil_matrix((3*(N-1), 3*(N-1)))

        # N-1 viscous torque equations.
        A[0: N-1, 0:N-1], A[0: N-1, N-1:2*(N-1)], A[0: N-1, 2*(N-1):] = self.viscous_torque_mat(x, y, theta)

        Cxx, Cxtheta, Cyy, Cytheta = self.kinematic_constraint_mat(theta)
        # N-2 kinematic constraints in x direction
        A[N-1:2*(N-1),0:N-1] = Cxx
        A[N-1:2*(N-1),2*(N-1):] = Cxtheta

        #N-2 kinematic constraints in y direction
        A[2*(N-1):,N-1:2*(N-1)] = Cyy
        A[2*(N-1):,2*(N-1):] = Cytheta
        return A.tocsr()

    def rhs(self, theta, t):
        N = self.N
        theta = append(zeros(1),theta)
        spring_const = self.spring_const

        f = spring_const*self.C_distribution* diff(theta)
        f = append(f, zeros(2*(N-1)))
        f[2*(N-1)] = -self.f_actuation(t) #ydot2, kinematic constraint
        f[0] = f[0] + (-self.f_actuation(t))*(-(1 / 2) * (1 / N ** 2) * cos(theta[1]))
        return f

    def dydt(self, y,t):
        """
        y' = func(y,t)
        y = x,y,theta
        """
        print('time = {}'.format(t))
        self.y2 = self.f_y1(t)
        N = self.N
        xvec = y[0:N-1]
        yvec = y[N-1:2*(N-1)]
        thetavec = y[2*(N-1):]
        A = self.lhs(xvec,yvec,thetavec)
        r = self.rhs(thetavec, t)
        yp = spsolve(A, r, use_umfpack = True)
        self.xdot[self.dotindex, :] = yp[0:N-1]
        self.ydot[self.dotindex, :] = yp[N - 1:2*(N-1)]
        self.thetadot[self.dotindex, :] = yp[2*(N-1):3*(N-1)]
        self.dotindex += 1
        return yp

    def run(self, tvals, saving=True):
        # time the program
        realtime0 = time.time()

        N = self.N
        xi = linspace(2 / N, 1, N - 1)
        yi = self.f_y1(0) * ones(N - 1)
        thetai = zeros(N - 1)

        self.xdot = zeros((len(tvals),N-1))
        self.ydot = zeros((len(tvals),N-1))
        self.thetadot = zeros((len(tvals),N-1))
        self.dotindex = 0

        # initial conditions.
        y0 = append(append(xi, yi), thetai)

        sol, info = odeint(self.dydt, y0, tvals, args=(), full_output=True, \
                           rtol=1.0e-9, atol=1.0e-9, mxstep=1000)
        print(info['message'])

        # extract solution from the solution array.
        x = sol[:,0:N-1]
        y = sol[:, N-1:2*(N-1)]
        theta = sol[:,2*(N-1):]

        #Calculate force
        Fx_coef_xdot, Fx_coef_y_dot, Fx_coef_theta_dot = self.viscous_force_x_coef(theta)
        Fx2 = Fx_coef_y_dot[:,0]*self.f_actuation(tvals) + Fx_coef_theta_dot[:,0]*self.thetadot[:,0]
        Fx = Fx_coef_xdot[:,1:]*self.xdot[:,:-1] + Fx_coef_y_dot[:,1:]*self.ydot[:,:-1] + Fx_coef_theta_dot[:,1:]*self.thetadot[:,1:]
        Fx = sum(Fx, axis = 1) + Fx2

        #Prepare vectors
        zerovec = zeros((len(tvals),1))
        x2vec = 1/N*ones((len(x),1))
        x = concatenate((zerovec,x2vec,x), axis = 1)

        y1 = self.f_y1(tvals)
        y1 = y1[np.newaxis]
        y = concatenate((y1.T,y1.T,y), axis = 1)

        theta1vec = zeros((len(x),1))
        theta = concatenate((theta1vec, theta), axis=1)

        if saving:
            dict = {}
            # convert to float for possible use in matlab,
            # otherwise 1/N=0  if N>1 in matlab if N is int.
            dict['N'] = float(N)
            dict['msg'] = info['message']
            dict['t'] = tvals
            dict['x'] = x
            dict['y'] = y
            dict['theta'] = theta
            dict['gamma'] = float(self.gamma)
            dict['spring_const'] = float(self.spring_const)
            dict['sp'] = float(self.sp)
            dict['C_distribution'] = self.C_distribution
            dict['xdot'] = self.xdot[:,:-1]
            dict['ydot'] = self.ydot[:,:-1]
            dict['thetadot'] = self.thetadot
            dict['Fx'] = Fx
            dict['dt'] = self.dt
            dict['exponent'] = self.exponent
            fname = 'sp{:.2f}N{:d}dt10e-{:d}.mat'.format(self.sp, self.N, self.exponent)
            savemat(fname, dict)
        realtime1 = time.time()
        elapsedtime = realtime1 - realtime0
        timefmt = time.strftime('%H hours %M minutes and %S seconds', time.gmtime(elapsedtime))
        print('Simulation took {}.'.format(timefmt))
        return sol, info

if __name__ == "__main__":
    epsilon = 0.5
    f_actuation = lambda t: -epsilon * sin(t)
    f_y1 = lambda t: epsilon * cos(t)
    stiffness_type = 'linear'
    stiffness_params = [-1,1]
    sim = Clamped(Nlink=5, gamma=1.2, sp=0.5, epsilon = epsilon, f_y1 = f_y1, f_actuation = f_actuation)
    sim.stiffness_distribution(stiffness_type, stiffness_params)
    exponent = 5
    dt = 1*10**-exponent
    sim.dt = dt
    sim.exponent = exponent
    tvals = arange(0, 20 + dt, dt)
    sim.run(tvals, saving=True)