"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5                   # [kg]
GRAVITY = -9.81                       # [m/s^2]
MOI = np.array([0.005, 0.005, 0.01])  # [kg * m^2]
MAX_THRUST = 10.0                     # [N]
MAX_TORQUE = 1.0                      # [N * m]
EPSILON = 1.0E-4
MAX_TILT = 1.0

class PDController(object):
    def __init__(self, k_p, k_d):
        self.k_p = k_p
        self.k_d = k_d

    def control(self, error, error_dot, feed_forward=0.0):
        return self.k_p * error +  self.k_d * error_dot + feed_forward

class PController(PDController):
    def __init__(self, k_p):
        super().__init__(k_p=k_p, k_d=0.0)

    def control(self, error):
        return super().control(error, error_dot=0.0, feed_forward=0.0)

def normalize_angle(x):
    """Normalize angle to the range [pi, -pi]."""
    x = (x + np.pi) % (2.0*np.pi)

    if x < 0:
        x += 2.0*np.pi

    return x - np.pi

class NonlinearController(object):
    def __init__(self):
        """Initialize the controller object and control gains"""
        # Altitude controller (PD controller)
        self.altitude_controller_ = PDController(k_p=8.0, k_d=4.0)

        # Yaw controller (P controller)
        self.yaw_controller_ = PController(k_p=5.5)

        # Body-rate controller (P controllers)
        pq_controller = PController(k_p=20.0)
        self.p_controller_ = pq_controller
        self.q_controller_ = pq_controller
        self.r_controller_ = PController(k_p=7.5)

        # Roll-pitch controller (P controllers)
        roll_pitch_controller = PController(k_p=8.0)
        self.roll_controller_ = roll_pitch_controller
        self.pitch_controller_ = roll_pitch_controller

        # Lateral controller (PD controllers)
        xy_controller = PDController(k_p=5.0, k_d=4.0)
        self.x_controller_ = xy_controller
        self.y_controller_ = xy_controller


    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]

        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)

        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                                 acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        acc_x = self.x_controller_.control(local_position_cmd[0] - local_position[0],
                                           local_velocity_cmd[0] - local_velocity[0],
                                           acceleration_ff[0])

        acc_y = self.y_controller_.control(local_position_cmd[1] - local_position[1],
                                           local_velocity_cmd[1] - local_velocity[1],
                                           acceleration_ff[1])

        return np.array([acc_x, acc_y])

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """
        thrust = 0.0

        R = euler2RM(*attitude)
        b_z = R[2][2]

        if abs(b_z) > EPSILON:
            error_z = altitude_cmd - altitude
            error_z_dot = vertical_velocity_cmd - vertical_velocity

            u_1_bar = self.altitude_controller_.control(error_z, error_z_dot, acceleration_ff)
            thrust = DRONE_MASS_KG * u_1_bar / b_z
            thrust = np.clip(thrust, 0.0, MAX_THRUST)
        else:
            print('b_z = 0.0, cannot compute thrust!')

        return thrust

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thrust command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        roll_pitch_rate_cmd = np.array([0.0, 0.0])

        if abs(thrust_cmd) > EPSILON:
            R = euler2RM(*attitude)

            if abs(R[2][2]) > EPSILON:
                # Current attitude
                b_a_x = R[0,2]
                b_a_y = R[1,2]

                # Desired attitude
                # Thrust comes with positive up, but in NED it should be positive down!
                # Also, b_* must be dimensionless so convert thrust to acceleration
                b_c_x = acceleration_cmd[0] / (-thrust_cmd / DRONE_MASS_KG)
                b_c_y = acceleration_cmd[1] / (-thrust_cmd / DRONE_MASS_KG)

                # Clip desired attitude to ensure the drone won't go upside down
                b_c_x = np.clip(b_c_x, -MAX_TILT, MAX_TILT)
                b_c_y = np.clip(b_c_y, -MAX_TILT, MAX_TILT)

                # Compute desired roll and pitch rates in world frame
                b_c_x_dot = self.roll_controller_.control(b_c_x - b_a_x)
                b_c_y_dot = self.pitch_controller_.control(b_c_y - b_a_y)

                # Convert to body frame
                M = np.array([[R[1,0], -R[0,0]],
                              [R[1,1], -R[0,1]]])
                b_c_dot = np.array([b_c_x_dot, b_c_y_dot])

                roll_pitch_rate_cmd = (1.0 / R[2,2]) * np.matmul(M, b_c_dot)
            else:
                print('R[2][2] = 0.0, cannot compute roll_pitch_rate!')
        else:
            print('thrust_cmd = 0.0, cannot compute roll_pitch_rate!')

        return roll_pitch_rate_cmd

    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        moment_p = MOI[0] * self.p_controller_.control(body_rate_cmd[0] - body_rate[0])
        moment_q = MOI[1] * self.q_controller_.control(body_rate_cmd[1] - body_rate[1])
        moment_r = MOI[2] * self.r_controller_.control(body_rate_cmd[2] - body_rate[2])

        moment_p = np.clip(moment_p, -MAX_TORQUE, MAX_TORQUE)
        moment_q = np.clip(moment_q, -MAX_TORQUE, MAX_TORQUE)
        moment_r = np.clip(moment_r, -MAX_TORQUE, MAX_TORQUE)

        return np.array([moment_p, moment_q, moment_r])

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        error = normalize_angle(yaw_cmd - yaw)
        return self.yaw_controller_.control(error)
