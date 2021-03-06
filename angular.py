from numpy import sin, cos, linspace, ones, zeros, arange, exp, sum
from numpy import append, diff, pi
from numpy import concatenate
from scipy.io import savemat
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import math
#ode integrator
from scipy.integrate import odeint
import time

from angular_multilink import Multilink

class Angular(Multilink):
    """
    Class for the clamped case

    Attributes
    ----------
    :param Nlink : int
        Number of links.
    :param gamma : float
        Resistance ratio from RFT, gamma >1.
    :param spring_const : float
        Dimensionless spring constant.
    :param sp : float
        Sperm number.
    :param epsilon : float
        Dimensionless oscillation amplitude
    :param f_theta1: anonymous function
        angle of first filament
    :param f_thetadot1: anonymous function
        angular velocity of first filament
    :param f_x2: anonymous function
        x position of second filament
    :param f_xdot2: anonymous function
        x velocity of second filament
    :param f_y2: anonymous function
        y position of second filament
    :param f_ydot2: anonymous function
        y velocity of second filament

    """
    def __init__(self, Nlink, gamma, sp, BigTheta):
        super(Angular, self).__init__(Nlink, gamma, sp)
        self.BigTheta = BigTheta

    def stiffness_distribution(self, stiffness_type, stiffness_params):
        """
        Creates stiffness distribution, C_distribution, based on stiffness type and parameters
        C_distribution is saved to self

        :param stiffness_type: str
            stiffness distribution description (linear, quadratic, exponential)
        :param stiffness_params: list
            Spring distribution parameters
        :return: None
        """
        N = self.N
        self.spring_positions = linspace(1/N,1-1/N,N-1)
        if stiffness_type == 'linear':
            a, b = stiffness_params
            C = a*self.spring_positions+b
            self.C_distribution = C
        elif stiffness_type == 'quadratic':
            a,b,c = stiffness_params
            C = a*self.spring_positions**2 + b*self.spring_positions + c
            self.C_distribution = C
        elif stiffness_type == 'exponential':
            a = stiffness_params
            C = exp(a*self.spring_positions)
            self.C_distribution = C
        else:
            raise ValueError(
                'spring distribution should be one of linear, quadratic, or exponential, while the input given is: %s' % stiffness_type)

        if min(self.C_distribution) < 0:
            raise ValueError(
                'spring distirbution should have positive values, while the minimum value of the stiffness distribution is: %.1f' % min(self.C_distribution)
            )
        return

    def actuation_conditions(self, angular_actuation_function):
        """
        Get actuation function anonymous functions

        :param angular_actuation_function:
        :return:

        :param f_theta1: anonymous function
            angle of first filament
        :param f_thetadot1: anonymous function
            angular velocity of first filament
        :param f_x2: anonymous function
            x position of second filament
        :param f_xdot2: anonymous function
            x velocity of second filament
        :param f_y2: anonymous function
            y position of second filament
        :param f_ydot2: anonymous function
            y velocity of second filament
        """

        if angular_actuation_function == 'cos':
            self.f_thetadot1 = lambda t: -BigTheta * sin(t)
            self.f_theta1 = lambda t: BigTheta * cos(t)
            self.f_x2 = lambda theta1: 1 / N * cos(theta1)
            self.f_xdot2 = lambda t, theta1: 1 / N * BigTheta * sin(t) * sin(theta1)
            self.f_y2 = lambda theta1: 1 / N * sin(theta1)
            self.f_ydot2 = lambda t, theta1: - 1 / N * BigTheta * sin(t) * cos(theta1)
        elif angular_actuation_function == 'sin':
            self.f_thetadot1 = lambda t: BigTheta * cos(t)
            self.f_theta1 = lambda t: BigTheta * sin(t)
            self.f_x2 = lambda theta1: 1 / N * cos(theta1)
            self.f_xdot2 = lambda t, theta1: - 1 / N * BigTheta * cos(t) * sin(theta1)
            self.f_y2 = lambda theta1: 1 / N * sin(theta1)
            self.f_ydot2 = lambda t, theta1: 1 / N * BigTheta * cos(t) * cos(theta1)
        else:
            raise ValueError(
                "actuation function should be one of 'cos' or 'sin,' while the input is: %s" \
                % angular_actuation_function)


    def lhs(self, x, y, theta):
        """
        :param x: numpy.ndarray
            array of x positions of filaments 3 throuh N + 1 at current time step
        :param y: numpy.ndarray
            array of y positions of filaments 3 throuh N + 1 at current time step
        :param theta: numpy.ndarray
            array of angles of filaments 2 throuh N at current time step
        :return Compressed Sparse Row
            3(N-1) x 3(N-1) matrix of linear equations, described below

        The system is represented by AX = b, where
        X is a 3(N-1) x 1 column vector
        A is a 3(N-1) x 3(N-1) square matrix
        b is a 3(N-1) x 1 column vector

        X = [x3, x4, ..., xN, xN+1, y3, y4, ..., yN, yN+1, theta2, theta3, ..., thetaN].
        b is created in self.rhs

        A is a 3(N-1) x 3(N-1) square matrix
        The first (N-1) entries represent torque balances
        eq[0]: torque balance of last N-1 links evaluated at x2
        eq[1]: torque balance of last N-2 links evaluated at x3
        eq[2]: torque balance of last N-3 links evaluated at x4
        ...
        ...
        ...
        eq[N-3]: torque balance of the two links evaluated at xN-1
        eq[N-2]: torque balance of the last link evaluated at xN

        The second set of N+1 entries represent kinematic constraints in the x axis
        eq[N-1]: -xdot3 - 1/N*thetadot2*sin(theta2) = -1/N*BigTheta*sin(t)*sin(theta1)
        eq[N]:  xdot3 - xdot4 - 1/N*thetadot3*sin(theta3) = 0
        ...
        ...
        ...
        eq[2(N-1)-1]: xdotN - xdot_{N+1} - 1/N*thetadotN*sin(thetaN) = 0

        The last set of N+1 entries represent kinematic constraints in the y axis
        eq[2(N-1)]: - ydot3 + 1/N*thetadot2*cos(theta2) = 1/N*BigTheta*sin(t)*cos(theta2)
        eq[2(N-1)+1]: ydot3 - ydot4 +1/N*thetadot3*cos(theta3) = 0
        ...
        ...
        ...
        eq[3(N-1)-1]: ydot_[N] - ydot_[N+1] - 1/N*thetadot_[N]*cos(theta_[N]) = 0

        """
        N = self.N
        A = lil_matrix((3*(N-1), 3*(N-1)))

        # N-1 viscous torque equations.
        A[0: N-1, 0:N-1], A[0: N-1, N-1:2*(N-1)], A[0: N-1, 2*(N-1):] = self.viscous_torque_mat(x, y, theta)

        Cxx, Cxtheta, Cyy, Cytheta = self.kinematic_constraint_mat(theta)
        # N-1 kinematic constraints in x direction
        A[N-1:2*(N-1),0:N-1] = Cxx
        A[N-1:2*(N-1),2*(N-1):] = Cxtheta

        # N-1 kinematic constraints in y direction
        A[2*(N-1):,N-1:2*(N-1)] = Cyy
        A[2*(N-1):,2*(N-1):] = Cytheta
        return A.tocsr()

    def rhs(self, theta, t):
        """
        Creates RHS vector, b

        Input:
        :param theta: numpy.ndarray
            angles of each filament (1 through N)
        :param t: float
            time of current step
        :return f:
            3(N-1) x 1 rhs vector

        f is a 3(N-1) x 1 vector
        The first N-1 entries represent the rhs of torque balances
        entry[0] = k2*(theta2-theta1) - T_22x*xdot2 - T_22y*ydot2
        entry[1] = k3*(theta3 - theta2)
        ...
        ...
        ...
        entry[N-3]: kN-1*(thetaN-1 - thetaN-2)
        entry[N-2]: kN*(thetaN - thetaN-1)

        The second N-1 entries represent the rhs of the kinematic constraints in the x direction
        All entries are zero, except for N-1
        entry[N-1]: -1/N*sin(t)*sin(theta1) = -x2
        entry[N]: 0
        ...
        ...
        ...
        entry[2(N-1)-2]: 0
        entry[2(N-1)-1]: 0

        The third and final N entries represent the rhs of the kinematic constraints in the y direction
        All entries are zero, except for entry 2(N-1)
        entry[2(N-1)]: 1/N*sin(t)*cos(theta1) = -y2
        entry[2(N-1)+1]: 0
        entry[2(N-1)+2]: 0
        ...
        ...
        ...
        entry[3(N-1)-2]: 0
        entry[3(N-1)-1]: 0
        """

        N = self.N
        theta1 = self.theta1
        theta2 = theta[0]
        x2 = self.f_xdot2(t,theta1)
        y2 = self.f_ydot2(t,theta1)
        theta = append(theta1,theta)
        spring_const = self.spring_const

        f = spring_const*self.C_distribution* diff(theta)
        f = append(f, zeros(2*(N-1)))
        f[(N - 1)] = -x2 # xdot2, kinematic constraint
        f[2*(N-1)] = -y2 # ydot2, kinematic constraint
        f[0] = f[0] - x2*(1/2*(1/N)**2*sin(theta2)) - y2*(-1/2*(1/N)**2*cos(theta2))
        return f

    def dydt(self, y,t):
        """
        Solves for xdot, ydot, and thetadot at the given time step
        yp = [xdot, ydot, thetadot]

        Saves xdot, ydot, and thetadot to self for future use outside of function
        Returns yp so that odeint function can solve for x, y, and theta

        self.dotindex: int
            Index to keep track of current time step
            Start at zero (0)
            Increases for each time step

        :param y: numpy.ndarray
            array containing x, y, and theta
            x positions for links 3 through N+1 [0:N-1]
            y positions for links 3 through N+1 [N-1:2*(N-1)]
            angles theta for links 2 through N  [2*(N-1):]
        :param t: float
            time at current time step

        :return yp: numpy.ndarray
            solution X, where AX = b
            X is an array of xdot, ydot, thetadot
            xdot for links 3 through N+1 [0:N-1]
            ydot for links 3 through N+1 [N-1:2*(N-1)]
            thetadot for links 2 through N [2*(N-1):]
        """
        print('time = {}'.format(t))
        N = self.N
        xvec = y[0:N - 1]
        yvec = y[N - 1:2 * (N - 1)]
        thetavec = y[2 * (N - 1):]

        # Get theta1, x2, and y2 at current time step
        self.theta1 = self.f_theta1(t)
        self.x2 = self.f_x2(self.theta1)
        self.y2 = self.f_y2(self.theta1)

        # Compute A and r
        A = self.lhs(xvec,yvec,thetavec)
        r = self.rhs(thetavec, t)

        # Solve system A*yp = r
        yp = spsolve(A, r, use_umfpack = True)

        # Save entries of yp (xdot, ydot, and thetadot)
        # Use current dotindex
        self.xdot[self.dotindex, :] = yp[0:N-1]
        self.ydot[self.dotindex, :] = yp[N - 1:2*(N-1)]
        self.thetadot[self.dotindex, :] = yp[2*(N-1):3*(N-1)]

        # Step dotindex
        self.dotindex += 1

        return yp

    def force_x(self, theta):
        """
        Calculates the force in the x direction
        The force acting on the first filament is due only to angular velocity
        We have a formula for the values of xdot2 and ydot2
        We have previously solved for thetadot2
        We have all of xdot, ydot, and thetadot for 3 through N

        :param theta: numpy.ndarray
            array of angles 2 through N
        :return Fx: numpy.ndarray
            Force acting in x direction at each time step
        """
        thetaLength = theta.shape[1]
        if thetaLength != self.N:
            raise ValueError(
                'array theta should be of length {}, while the input is of length {}' .format(self.N,thetaLength))

        Fx_coef_x_dot, Fx_coef_y_dot, Fx_coef_theta_dot = self.viscous_force_x_coef(theta)
        Fx1 = Fx_coef_theta_dot[:,0]*self.thetadot1
        Fx2 = Fx_coef_x_dot[:,1]*self.xdot2 + Fx_coef_y_dot[:,1]*self.ydot2 + Fx_coef_theta_dot[:,1]*self.thetadot[:,0]
        Fx = Fx_coef_x_dot[:,2:]*self.xdot[:,:-1] + Fx_coef_y_dot[:,2:]*self.ydot[:,:-1] + Fx_coef_theta_dot[:,2:]*self.thetadot[:,1:]
        return sum(Fx, axis=1) + Fx1 + Fx2

    def Xdot_init(self, tvals):
        """
        Initialize xdot, ydot, and thetdot matrices as zeros matrices
        Start dotindex at 0
        :param: tvals
        :return: None
        """
        N = self.N
        self.xdot = zeros((len(tvals),N-1))
        self.ydot = zeros((len(tvals),N-1))
        self.thetadot = zeros((len(tvals),N-1))
        self.dotindex = 0
        self.xdot2 = self.f_xdot2(tvals, self.f_theta1(tvals))
        self.ydot2 = self.f_ydot2(tvals, self.f_theta1(tvals))
        return

    def xy_vector_prep(self, x,y,theta):
        """
        Adds columns x1 = 0, x2 = 1/N
        Adds columns y1 = y2 = f_y1
        Adds column theta1 = 0

        :param x: numpy.ndarray
            x positions of links 3 through N+1 at each time step
        :param y: numpy.ndarray
            y positions of links 3 through N+1 at each time step
        :param theta: numpy.ndarray
            angles of links 2 through N at each time step
        :return x: numpy.ndarray
            x positions of links 1 through N+1 at each time step
        :return y: numpy.ndarray
            y positions of links 1 through N+1 at each time step
        :return theta: numpy.ndarray
            angles of links 1 through N+1 at each time step
        """
        theta1 = self.f_theta1(tvals)[np.newaxis].T
        x2 = self.f_x2(theta1)
        y2 = self.f_y2(theta1)

        zerovec = zeros((len(tvals), 1)) #for x1 and y1

        x = concatenate((zerovec, x2, x), axis=1)
        y = concatenate((zerovec,y2,y), axis = 1)
        theta = concatenate((theta1, theta), axis=1)
        return x, y, theta

    def spName(self):
        """
        Creates sp file name
        :return: fname: str
        """
        sp = self.sp

        if sp >= 1 and sp == int(sp):
            fname = '{:d}_00'.format(int(sp))
            return fname
        elif sp >= 1 and sp != int(sp):
            fname = '{:d}_'.format(int(sp))
            sp = math.ceil((sp - int(sp)) * 10000)
            fname = fname + '{:d}'.format(sp)
            fname = fname[:-2]
            return fname
        elif sp < 1:
            fname = '0_'
            sp = math.ceil(sp * 10000)
            fname = fname + '{:d}'.format(sp)
            fname = fname[:-2]
            return fname
        return

    def run(self, tvals, saving=True):
        """
        Runs simulation over timesteps tvals
        Finds xdot, ydot, and thetadot
        Find x, y, and theta
        Finds Fx
        Saves data to a file

        :param tvals: numpy.ndarray
            array of tvals from time zero to time end
            with time steps of time dt
        :param saving: bool
            indicates whether or not to save data (will typically be True)
        :return sol: numpy.ndarray
            array containing x, y, theta
        :return info: str
            output message from odeint function

        Attributes:
            y0: numpy.ndarray
                Composed of initial conditions xi, yi, and thetai, y0 = [xi,yi,thetai]
                xi [0:N], yi [N:2*N], thetai [2*N:]
            xdot, ydot, and thetadot: numpy.ndarray
                matrices containing xdot, ydot, and thetadot at each time step
            x, y, and theta: numpy.ndarray
                extracted solution at each time step
        """

        # time the program
        realtime0 = time.time()

        BC = 'angular'
        N = self.N

        # Initial conditions for x, y, and theta
        theta0 = self.f_theta1(0)
        thetai = theta0 * ones(N - 1)
        xi = zeros(N+1)
        yi = zeros(N+1)
        for idx in range(0,N):
            xi[idx+1] = xi[idx] + 1/N*cos(theta0)
            yi[idx+1] = yi[idx] + 1/N*sin(theta0)

        xi, yi = xi[2:], yi[2:]

        y0 = append(append(xi, yi), thetai)

        self.Xdot_init(tvals)

        sol, info = odeint(self.dydt, y0, tvals, args=(), full_output=True, \
                           rtol=1.0e-9, atol=1.0e-9, mxstep=1000)
        print(info['message'])

        # extract solution from the solution array.
        x = sol[:,0:N-1]
        y = sol[:, N-1:2*(N-1)]
        theta = sol[:,2*(N-1):]

        # Create vectors for theta1, x2, and y2


        # Calculate force in x direction and new x and y position vectors
        x, y, theta = self.xy_vector_prep(x, y, theta)
        self.thetadot1 = self.f_thetadot1(tvals)
        Fx = self.force_x(theta)

        if saving:
            dict = {}
            # convert to float for possible use in matlab,
            # otherwise 1/N=0  if N>1 in matlab if N is int.
            dict['BC'] = BC
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
            dict['xdot'] = self.xdot[:, :-1]
            dict['ydot'] = self.ydot[:, :-1]
            dict['thetadot'] = self.thetadot
            dict['Fx'] = Fx
            dict['dt'] = self.dt
            fname = BC + '-sp' + self.spName() + 'N{:d}dt{dt}'.format(N, dt='%.E' % dt)
            dict['fname'] = fname
            fname = fname + '.mat'
            savemat(fname, dict)
        realtime1 = time.time()
        elapsedtime = realtime1 - realtime0
        timefmt = time.strftime('%H hours %M minutes and %S seconds', time.gmtime(elapsedtime))
        print('Simulation took {}.'.format(timefmt))
        return sol, info

if __name__ == "__main__":
    BigTheta = pi/4
    N = 5
    angular_actuation_function = 'sin' # cos or sin
    stiffness_type = 'linear'
    stiffness_params = [-1,1]
    sim = Angular(Nlink=N, gamma=1.2, sp=2.00, BigTheta = BigTheta)
    sim.stiffness_distribution(stiffness_type, stiffness_params)
    sim.actuation_conditions(angular_actuation_function)
    dt = 1e-3
    sim.dt = dt
    tvals = arange(0, 6*pi + dt, dt)
    sim.run(tvals, saving=True)