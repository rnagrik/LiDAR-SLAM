import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

class ICP:
    '''
    Iterative Closest Point (ICP) algorithm to align two point clouds
    Inputs:
        data_1: m x n array, of source point cloud with m dimensions and n points
        data_2: m x p array, of target point cloud with m dimensions and p points
        T_guess: (m+1) x (m+1) array, initial guess for transformation matrix
        convergence_error: float, convergence criteria based on change in error
        max_iter: int, maximum number of iterations
        fit_data_1_to_2: bool, if True, fit data_1 to data_2 regardless of number of points
    '''

    def __init__(self, data_1, data_2, T_guess, convergence_error=10e-3, max_iter=50, fit_data_1_to_2=False):        

        # Set the source and target scans based on number of points
        if data_1.shape[0] <= data_2.shape[0] or fit_data_1_to_2:
            self.target_scan = data_2
            self.source_scan = data_1
            self.inverse = False
        else:
            self.target_scan = data_1
            self.source_scan = data_2
            self.inverse = True
            
        self.dim = np.min(data_1.shape)
        self.R = T_guess[:-1,:-1]                            # initial rotation guess
        self.p = T_guess[:-1,-1].reshape((self.dim,1))       # initial translation guess
        
        self.max_iter = max_iter
        self.error = None
        self.convergence_error = convergence_error

        # z and m correspondence, larger data frame (m) is considered as reference frame
        self.z = self.source_scan                            # source points
        self.m = None                                        # target points corresponding to source points
        self.count_z = self.z.shape[1]                  
        self.search_tree = KDTree(self.target_scan.T)        # build a KDTree for faster nearest neighbor search
        self.FindCorrespondenceAndError()                    # get initial z-m correspondence and error
        self.prev_error = self.error

        self.FinalTransformedPoints = None
        self.FinalTransformation = np.eye(self.dim+1)       # final T matrix from source to target frame

    def FindCorrespondenceAndError(self):
        # find the correspondence and error given current R and p
        transformed_data = np.matmul(self.R, self.source_scan) + self.p
        dist, indices = self.search_tree.query(transformed_data.T)
        self.error = sum(dist)
        self.m = self.target_scan[:,indices]

    def performKabsch(self):
        # Perform Kabsch algorithm to find optimal R and p, given z-m correspondence
        z_avg = np.sum(self.z,axis=1).reshape((self.dim,1))/self.count_z
        m_avg = np.sum(self.m,axis=1).reshape((self.dim,1))/self.count_z
        
        del_z = self.z - z_avg
        del_m = self.m - m_avg
        Q = np.matmul(del_m,del_z.T)
        U,S,VT = np.linalg.svd(Q)
        S_star = np.eye(self.dim)
        S_star[self.dim-1,self.dim-1] = np.linalg.det(np.matmul(U,VT))

        self.R = np.matmul(np.matmul(U,S_star),VT)
        self.p = m_avg - np.matmul(self.R,z_avg)

    def startIterations(self):
        # Iteratively perform Kabsch and find correspondence until convergence or max iterations
        err_conv_count = 0                         # to count number of consecutive iterations with small error change
        for iter_count in range(self.max_iter):
            self.performKabsch()                   # Get optimal R and p using Kabsch algorithm
            self.FindCorrespondenceAndError()      # Find new correspondence and error
            # print(iter_count, self.error)
            iter_count += 1

            if abs((self.prev_error - self.error)/self.prev_error) < self.convergence_error:
                err_conv_count += 1
            else:
                err_conv_count = 0

            self.prev_error = self.error
            if err_conv_count > 5:
                break

        # Get final transformed points and transformation matrix, invert if needed
        if self.inverse:
            self.FinalTransformedPoints = np.matmul(self.R.T,self.target_scan) - self.p
            self.FinalTransformation[:-1,:-1] = self.R.T
            self.FinalTransformation[:-1,-1] = -self.p.T
        else:
            self.FinalTransformedPoints = np.matmul(self.R,self.source_scan) + self.p
            self.FinalTransformation[:-1,:-1] = self.R
            self.FinalTransformation[:-1,-1] = self.p.T
        

class LiDARDataHandler:
    '''
    Handle LiDAR data and convert to x,y coordinates in LiDAR frame
    inputs:
        lidar_data:  m x n array, of LiDAR range data for n scans with m points each
        lidar_info: dictionary, with LiDAR specifications
            angle_max, angle_min, angle_increment, range_max, range_min
    '''
    def __init__(self,lidar_data,lidar_info):
        self.lidar_data = lidar_data
        self.n_instances = lidar_data.shape[1]    
        self.lidar_angle_max = np.float64(lidar_info['angle_max'])
        self.lidar_angle_min = np.float64(lidar_info['angle_min'])
        self.lidar_angle_increment = np.float64(lidar_info['angle_increment'])
        self.lidar_range_max = np.float64(lidar_info['range_max'])
        self.lidar_range_min = np.float64(lidar_info['range_min'])
        
        # to stay consistent with the expected size of the data, lidar_truth is introduced
        self.lidar_truth = None               # boolean array (mask) to identify valid points
        self.lidar_data_filtered = None       # filtered lidar data as per range limits
        self.lidar_xy = None                  # x,y coordinates of lidar points in lidar frame
        self.conv2cartesian()

    def conv2cartesian(self):
        # Filter out invalid LiDAR points based on range limits and convert to x,y coordinates in LiDAR frame
        truth_high = self.lidar_data < self.lidar_range_max
        truth_low = self.lidar_data > self.lidar_range_min
        self.lidar_truth = truth_high * truth_low                       # boolean array to identify points within range limits
        self.lidar_data_filtered = self.lidar_data * self.lidar_truth   # filter out invalid points based on range limits

        # Convert filtered LiDAR data to polar coordinates
        angles = np.linspace(self.lidar_angle_min,self.lidar_angle_max,int((self.lidar_angle_max-self.lidar_angle_min)/self.lidar_angle_increment)+1).reshape((-1,1))
        angles = np.tile(angles,self.n_instances)

        # Calculate x,y coordinates in LiDAR frame, x = r*cos(theta), y = r*sin(theta)
        lidar_x = np.cos(angles) * self.lidar_data_filtered       # shape m x n
        lidar_y = np.sin(angles) * self.lidar_data_filtered       # shape m x n
        self.lidar_xy = np.stack((lidar_x,lidar_y),axis=0)        # shape 2 x m x n

    

class LiDARICP:
    '''
    Using ICP, get realtive transformations and estimated trajectory from Lidar scans
    Inputs:
        lidar_xy:    2 x m x n array, of LiDAR points in LiDAR frame (x,y) for n scans with m points each
        lidar_truth: m x n boolean array, to identify valid points from LiDAR data for n scans with m points each
        initial_icp: (n-1) x 3 x 3 array, of initial guess for ICP
    '''

    def __init__(self,lidar_xy,lidar_truth,initial_icp):
        self.data = lidar_xy
        self.lidar_truth = lidar_truth                # boolean array to identify valid points from LiDAR data
        self.initial_icp = initial_icp                # initial guess for ICP from odometry
        self.data_size = self.data.shape[-1]        
        self.state = np.array([[0],[0],[0]])            # estimated robot states (x,y,theta)
        self.all_rot_mat = np.eye(3).reshape((1,3,3))   # all transformations
        self.T_relative = np.eye(3).reshape((1,3,3))    # all relative transformations
        self.calculateTransforms()

    def calculateTransforms(self):

        # Iterate over all LiDAR scans and perform ICP between consecutive scans
        for i in tqdm(range(self.data_size-1)):
            # extract valid points for scan i
            data1_x = self.data[0,:,i][self.lidar_truth[:,i]]   
            data1_y = self.data[1,:,i][self.lidar_truth[:,i]]   
            data1 = np.concatenate((data1_x.reshape((1,-1)),data1_y.reshape((1,-1))),axis=0)
            # extract valid points for scan i+1
            data2_x = self.data[0,:,i+1][self.lidar_truth[:,i+1]] 
            data2_y = self.data[1,:,i+1][self.lidar_truth[:,i+1]] 
            data2 = np.concatenate((data2_x.reshape((1,-1)),data2_y.reshape((1,-1))),axis=0)
            # perform ICP between scan i and scan i+1
            transform = ICP(data1,data2,self.initial_icp[i,:,:],max_iter=100) 
            transform.startIterations()                                # perform ICP to get the tranformation

            T = transform.FinalTransformation                          # get the final T matrix from ICP
            self.T_relative = np.concatenate((self.T_relative,T.reshape(1,3,3)),axis=0)
            T = np.linalg.inv(T)                                       # invert to get T from scan i to scan i+1

            T_new = np.matmul(self.all_rot_mat[-1,:,:],T)              # calculate T from scan i+1 to world frame
            T_new = T_new.reshape((1,3,3))

            self.all_rot_mat = np.concatenate((self.all_rot_mat,T_new),axis=0)  # store the i+1 to world frame T

            new_state = [[T_new[0,0,-1]],
                         [T_new[0,1,-1]],
                         [np.arctan2(T_new[0].T[0,1],T_new[0].T[0,0])]]         # extract state (x,y,theta)

            self.state = np.hstack((self.state,new_state))                      # stack the new state

            
            # plt.scatter(data1[0,:],data1[1,:],c="red")
            # # plt.scatter(data2[0,:],data2[1,:])
            # # plt.show()

            # plt.scatter(transform.FinalTransformedPoints[0,:],transform.FinalTransformedPoints[1,:],c="green")
            # plt.scatter(data2[0,:],data2[1,:],c="blue")
            # plt.show()
        

    def PlotTrajectory(self,holdon=False):
        plt.plot(self.state[0,:],self.state[1,:],label='LiDAR ICP based Trajectory')
        if not holdon:
            plt.legend()
            plt.show()




def calcICPInitialGuess(odo_data, odo_ts, lidar_data, lidar_ts):
    '''
    Calculate initial guess for LiDAR-ICP using odometry data
    Inputs:
        odo_data: 3 x n array of odometry data (x, y, theta)
        odo_ts: 1 x n array of odometry ts
        lidar_data: 2 x m array of LiDAR data (x, y)
        lidar_ts: 1 x m array of LiDAR ts
    Outputs:
        MatchedICPInitializer: (m-1) x 3 x 3 array of initial guess for ICP
    '''

    MatchedICPInitializer = np.eye(3).reshape(1,3,3)
    MatchedOdo = np.zeros((3,1))
    
    # Match odometry data to LiDAR timestamps
    for lidar_instant in lidar_ts:
        diff_array = np.abs(odo_ts - lidar_instant)
        index = diff_array.argmin()
        MatchedOdo = np.hstack((MatchedOdo, odo_data[:,index].reshape((3,1))))
    
    # Calculate relative transformation between consecutive LiDAR scans from odometry data
    for i in range(lidar_data.shape[-1]-1):
        # get pose change in world frame
        del_p_world = (MatchedOdo[:2,i+1] - MatchedOdo[:2,i]).reshape((2,1))
        del_theta = MatchedOdo[2,i+1] - MatchedOdo[2,i]
        yaw = MatchedOdo[2,i]
        # convert pose change to robot frame
        rotmat =  np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        del_p_robot = np.matmul(rotmat.T,del_p_world)
        icp_rotmat = np.array([[np.cos(del_theta),-np.sin(del_theta)],[np.sin(del_theta),np.cos(del_theta)]])
        # construct transformation matrix 
        T_initial = np.eye(3)               # shape 3 x 3, for 2D ICP
        T_initial[:-1,:-1] = icp_rotmat.T
        T_initial[0,-1],T_initial[1,-1] = -del_p_robot[0,0], -del_p_robot[1,0]
        T_initial = T_initial.reshape((1,3,3))

        MatchedICPInitializer = np.concatenate((MatchedICPInitializer,T_initial),axis=0)

    return MatchedICPInitializer[1:,:,:] 


