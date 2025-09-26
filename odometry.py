import numpy as np
import matplotlib.pyplot as plt


class Odometry:
    '''
    Class to handle odometry data from IMU and wheel encoders
    Inputs:
    imu_data:       6 x n numpy array, of IMU data (3 linear acc, 3 angular vel)
    imu_ts:         n numpy array, of IMU timestamps
    encoder_data:   4 x n numpy array, of wheel encoder counts (FR, FL, RR, RL)
    encoder_ts:     n numpy array, of encoder timestamps
    wheel_radius:   float, radius of the wheels in meters (default 0.254/2 m)
    '''
    def __init__(self,imu_data,imu_ts,encoder_data,encoder_ts,wheel_radius=0.254/2):
        self.imu_data = imu_data
        self.imu_ts = imu_ts
        # encoder data order FR - FL - RR - RL
        self.encoder_data = encoder_data
        self.encoder_ts = encoder_ts
        self.dt = self.encoder_ts[1:]-self.encoder_ts[:-1]
        self.MatchDataTimestamps()

        self.r = wheel_radius
        self.all_states = None
        self.time_stamps = self.encoder_ts
        
        self.CalcStates()

    
    def MatchDataTimestamps(self):
        '''
        Match IMU data timestamps to encoder data timestamps for synchronized processing.
        '''
        e_count,i_count = 0, 0
        new_imu_data = np.zeros((self.imu_data.shape[0], self.encoder_ts.shape[0])) # new IMU data aligned to encoder timestamps
        
        while (e_count<self.encoder_ts.shape[0]) and (i_count<self.imu_ts.shape[0]-1):
            forward_diff = self.imu_ts[i_count+1]-self.encoder_ts[e_count]
            curr_diff = self.imu_ts[i_count]-self.encoder_ts[e_count]
            if abs(curr_diff) < abs(forward_diff):
                new_imu_data[:,e_count] = self.imu_data[:,i_count]
                e_count += 1
            i_count += 1
        e_count -= 1
        self.dt = self.dt[:e_count]
        self.encoder_ts = self.encoder_ts[:e_count]
        self.imu_ts = self.encoder_ts
        self.imu_data = new_imu_data[:,:e_count].T
        self.encoder_data = self.encoder_data[:,:e_count].T


    def CalcStates(self):
        '''
        Calculate the robot's trajectory (x, y, theta) using encoder and IMU data.
        x_t = x_(t-1) + v*cos(theta)*dt
        v_t = (r*(2*pi/360)/2)*(avg of all wheel encoders)/dt
        '''
        curr_state = np.array([[0,0,0]]).T
        all_states = curr_state
        t_count = 0

        while t_count < self.dt.shape[0]:
            v = ((np.sum(self.encoder_data[t_count,:])*(self.r*(2*np.pi/360)))/(2*2))/self.dt[t_count]
            w = self.imu_data[t_count,5] # yaw rate
            theta = curr_state[2,0]
            x_dot = np.array([[v*np.cos(theta),v*np.sin(theta),w]]).T
            new_state = curr_state + x_dot*self.dt[t_count]
            all_states = np.hstack((all_states,new_state))
            curr_state = new_state
            t_count +=1
        self.all_states = all_states[:,:-1]

    
    def PlotTrajectory(self,holdon=False):
        x_points = self.all_states[0,:]  
        y_points = self.all_states[1,:]
        theta = self.all_states[2,:]
        freq = 50
        fig, ax = plt.subplots()
        ax.plot(x_points, y_points,label="Encoder-IMU based Trajectory")
        for i in range(1, len(x_points), freq):
            ax.quiver(x_points[i], y_points[i], np.cos(theta[i]), np.sin(theta[i]), scale=50, color='red', width=0.005)
        if not holdon:
            plt.legend()
            plt.show()


