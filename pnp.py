import serial, time, argparse, ast, traceback
import numpy as np
from math import sin, cos, tan, degrees, radians, pi
from visualization import Network
from time import time

class Pose:
    def __init__(self, x, y, z, quat=None, euler=None, R_mat=None):
        assert not (quat and euler) # FIXME
        self._x = x
        self._y = y
        self._z = z
        self.quat, self.euler = None, None
        if quat:
            self.quat = quat
        elif euler:
            self.euler = euler
        self.R = R_mat

    def __repr__(self):
        return "(x: {}, y: {}, z: {}, R: {})".format(self.x(), self.y(), self.z(), self.R)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def theta_x(self):
        if self.euler:
            return self.euler[0]

    def theta_y(self):
        if self.euler:
            return self.euler[1]

    def theta_z(self):
        if self.euler:
            return self.euler[2]

    @staticmethod
    def rot_to_quat(R):
        return None # TODO: implement

    @staticmethod
    def rot_to_ypr(R): # http://nghiaho.com/?page_id=846
        theta_y = np.arctan2(-R[2][0], np.sqrt(R[2][1]**2 + R[2][2]**2))

        # degenerate case
        if np.abs(theta_y) == 90:
            sign = np.sign(theta_y)
            return np.arctan2(sign * R[1][2], sign * R[0][2]), theta_y, 0

        return degrees(np.arctan2(R[2][1], R[2][2])+pi), degrees(theta_y), degrees(np.arctan2(R[1][0], R[0][0]))

    @staticmethod
    def ypr_to_quat(y, p, r):
        qr = np.cos(r/2.)*np.cos(p/2.)*np.cos(y/2.) + np.sin(r/2.)*np.sin(p/2.)*np.sin(y/2.)
        qx = np.sin(r/2.)*np.cos(p/2.)*np.cos(y/2.) - np.cos(r/2.)*np.sin(p/2.)*np.sin(y/2.)
        qy = np.cos(r/2.)*np.sin(p/2.)*np.cos(y/2.) + np.sin(r/2.)*np.cos(p/2.)*np.sin(y/2.)
        qz = np.cos(r/2.)*np.cos(p/2.)*np.sin(y/2.) - np.sin(r/2.)*np.sin(p/2.)*np.cos(y/2.)

        return qx, qy, qz, qr

    # TODO: fill in quaternion methods

class Poser:
    NUM_CORR = 8

    def __init__(self, *args):
        self.diode_array = [loc for loc in args]

    def get_pose(self, angles):
        '''
        Triangulate 6-DoF pose of 4 diode 1 lighthouse system given
        azimuth and elevation angles of each diode.
        '''
        xy_n = [(tan(u), tan(v)) for u, v in angles]
        A = np.zeros((self.num_diodes()*2, Poser.NUM_CORR))
        b = np.array(xy_n).ravel()
        for r in range(0, self.num_diodes()*2, 2):
            i = int(r / 2)
            A[r] = [self.diode_x(i), self.diode_y(i), 1, 0, 0, 0, \
                        -self.diode_x(i)*xy_n[i][0], -self.diode_y(i)*xy_n[i][0]]
            A[r+1] = [0, 0, 0, self.diode_x(i), self.diode_y(i), 1, \
                        -self.diode_x(i)*xy_n[i][1], -self.diode_y(i)*xy_n[i][1]]

        try:
            # _h = np.linalg.solve(A, b)
            result = np.linalg.lstsq(A, b)
            _h = result[0]

            s = Poser.recover_scale(_h)
            x, y, z = s*_h[2], s*_h[5], -s
            R_mat = Poser.compute_rotation(_h)

            return Pose(x, y, z, R_mat=R_mat)
        except np.linalg.linalg.LinAlgError:
            estimator = LMOptimization(self, xy_n)
            estimator.compute_pose()
            return estimator.pose()

    def num_diodes(self):
        return len(self.diode_array)

    def diode_x(self, i):
        return self.diode(i)[0]

    def diode_y(self, i):
        return self.diode(i)[1]

    def diode(self, i):
        return self.diode_array[i]

    @staticmethod
    def recover_scale(h):
        return 2.0 / (np.sqrt(h[0]**2 + h[3]**2 + h[6]**2) + \
                        np.sqrt(h[1]**2 + h[4]**2 + h[7]**2))

    @staticmethod
    def compute_rotation(_h):
        h1, h2, h3, h4, h5, h6, h7, h8 = _h
        norm_factor = np.sqrt(h1**2 + h4**2 + h7**2)
        r11, r21, r31 = h1 / norm_factor, h4 / norm_factor, -h7 / norm_factor

        scale_factor = r11*h2 + r21*h5 - r31*h8
        _r2 = np.array([
                h2 - r11*scale_factor,
                h5 - r21*scale_factor,
                -h8 - r31*scale_factor
            ])

        r1 = np.array([r11, r21, r31])
        r2 = _r2 / np.linalg.norm(_r2)

        r3 = np.cross(r1, r2)
        R = np.column_stack((r1, r2, r3))

        if np.linalg.det(R) < 0:
            print("Negative determinant.")
            R[2][0] = -R[2][0]
            R[2][1] = -R[2][1]
            R[2][2] = -R[2][2]

        return R

class LMOptimization:
    def __init__(self, poser, angles, b,
                    p0=Pose(0, 0, -1, euler=(0.5, 0.5, 0.5)),
                    lda=1): # FIXME: play with best seed and lambda
        assert poser.num_diodes() == 4 # TODO: make optimization work for n > 4 lol
        self.poser = poser
        self.angles = angles
        self.b = b
        self.p = p0
        self.lda = lda

    def pose(self):
        return self.p

    def compute_pose(self, max_iter=100):
        for _ in range(max_iter):
            self.step()

    def step(self):
        f = self.evaluate_objective()
        J_g = self.compute_jacobian_g()
        J_f = self.compute_jacobian_f()

        print(J_g)
        print(J_f)

        J = np.dot(J_f, J_g)

        print(J)

        JTJ = np.dot(J.T, J)

        self.p += np.dot( np.linalg.inv(JTJ + self.lda*np.diagonal(JTJ)), self.b - f ) # dimensions are weird??

    def evaluate_objective(self):
        return np.zeros((8,)) # FIXME

    def evaluate_objective_error(self):
        result = 0
        h1, h2, h3, h4, h5, h6, h7, h8, h9 = self.g()
        for i in range(self.poser.num_diodes()):
            xn_i, yn_i = self.angles[i]
            x_i, y_i = self.poser.diode_x(i), self.poser.diode_y(i)

            f0 = (h1*x_i + h2*y_i + h3) / (h7*x_i + h8*y_i + h9)
            f1 = (h4*x_i + h5*y_i + h6) / (h7*x_i + h8*y_i + h9)

            result += (xn_i - f0)**2 + (yn_i - f1)**2
        return result

    def compute_jacobian_g(self): # do math by hand...
        if self.p.euler:
            x, y, z = self.p.euler
            return np.array([
                [-cos(x)*sin(y)*sin(z), -sin(y)*cos(z)-sin(x)*cos(y)*sin(z),
                    -cos(y)*sin(z)-sin(x)*sin(y)*cos(z), 0, 0, 0],
                [sin(x)*sin(z), 0, -cos(x)*cos(z), 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [cos(x)*sin(y)*cos(z), -sin(y)*sin(z)+sin(x)*cos(y)*cos(z),
                    -cos(y)*cos(z)-sin(x)*sin(y)*sin(z), 0, 0, 0],
                [-sin(x)*cos(z), 0, -cos(x)*sin(z), 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [-sin(x)*sin(y), cos(y)*cos(y),
                    -cos(y)*cos(z)-sin(x)*sin(y)*sin(z), 0, 0, 0],
                [-sin(x)*cos(z), 0, -cos(x)*sin(z), 0, 0, 0],
                [0, 0, 0, 0, 0, -1]
            ])
        elif self.p.quat:
            return None # FIXME: implement for quaternion representation
        else:
            return None # rotation matrix representation?

    def compute_jacobian_f(self): # do math by hand...
        h1, h2, h3, h4, h5, h6, h7, h8, h9 = self.g()
        rows = [
            np.array([
                [x_i / (h7*x_i + h8*y_i + h9), y_i / (h7*x_i + h8*y_i + h9), 1 / (h7*x_i + h8*y_i + h9), 0, 0, 0,
                    -(h1*x_i + h2*y_i + h3) / (h7*x_i + h8*y_i + h9)**2,
                    (h1*x_i + h2*y_i + h3) / (h7*x_i + h8*y_i + h9)**2,
                    -(h1*x_i + h2*y_i + h3) / (h7*x_i + h8*y_i + h9)**2],
                [0, 0, 0, x_i / (h7*x_i + h8*y_i + h9), y_i / (h7*x_i + h8*y_i + h9), 1 / (h7*x_i + h8*y_i + h9),
                    -(h4*x_i + h4*y_i + h6) / (h7*x_i + h8*y_i + h9)**2,
                    (h4*x_i + h4*y_i + h6) / (h7*x_i + h8*y_i + h9)**2,
                    -(h4*x_i + h4*y_i + h6) / (h7*x_i + h8*y_i + h9)**2]
            ])
            for x_i, y_i in self.poser.diode_array
        ]
        return np.vstack(tuple(rows))

    def g(self):
        x, y, z = self.p.theta_x(), self.p.theta_y(), self.p.theta_z()
        if self.p.euler:
            return np.array([
                cos(y)*cos(z) - sin(x)*sin(y)*sin(z),
                -cos(x)*sin(z),
                self.p.x(),
                cos(y)*sin(z) + sin(x)*sin(y)*cos(z),
                cos(x)*cos(z),
                self.p.y(),
                cos(x)*sin(y),
                -sin(x),
                -self.p.z()
            ])
        elif self.p.quat:
            return None # FIXME: implement for quaternion representation
        else:
            return None # rotation matrix representation?

class Kalman:
    def __init__(self, state, means, variances):
        self.state = state

def serial_data(port, baud):
    ser = serial.Serial(port, baud)
    val = ser.readline().decode("utf-8")
    ser.close()
    return val

if __name__ == "__main__":
    print("Building tracker...")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=str, help="The serial port.", default="/dev/cu.usbmodem49426301")
    parser.add_argument("-b", "--baud", type=int, help="The baud rate.", default=115200)
    args = parser.parse_args()

    x, y = .048, .0475
    tracker = Poser((-x, y), (x, y), (-x, -y), (x, -y)) # 2 more diodes

    _x, _y, _z, a, b, c = [], [], [], [], [], []
    i = 0.0

    # vis = Network()
    while i < 10000:
        #print(i)
        line = serial_data(args.port, args.baud)
        try:
            angles = ast.literal_eval(line)

            print(angles)

            # lighthouse 1
            tracking = (69.0, 69.0) not in angles[0]
            if tracking:
                pose = tracker.get_pose(angles[0])
                euler = Pose.rot_to_ypr(pose.R)
                print("Lighthouse 1: ", -pose.x(), -pose.y(), pose.z(), np.linalg.inv(pose.R))
                _x.append(pose.x())
                _y.append(pose.y())
                _z.append(pose.z())
                a.append(euler[0])
                b.append(euler[1])
                c.append(euler[2])
                i += 1
                # vis.update((pose.x(), pose.y(), -pose.z()), euler)
            else:
                print("Not tracking...")

            # lighthouse 2
            tracking = (69.0, 69.0) not in angles[1]
            if tracking:
                pose = tracker.get_pose(angles[1])
                euler = Pose.rot_to_ypr(pose.R)
                print("Lighthouse 2: ", -pose.x(), -pose.y(), pose.z(), np.linalg.inv(pose.R))
                _x.append(pose.x())
                _y.append(pose.y())
                _z.append(pose.z())
                a.append(euler[0])
                b.append(euler[1])
                c.append(euler[2])
                i += 1
                # vis.update((pose.x(), pose.y(), -pose.z()), euler)
            else:
                print("Not tracking...")
        except:
            traceback.print_exc()
            continue

    print("Variance pos:", np.var(_x), np.var(_y), np.var(_z))
    print("Mean pos:", np.mean(_x), np.mean(_y), np.mean(_z))

    print("Variance rot:", np.var(a), np.var(b), np.var(c))
    print("Mean rot:", np.mean(a), np.mean(b), np.mean(c))
    print("Done.")
