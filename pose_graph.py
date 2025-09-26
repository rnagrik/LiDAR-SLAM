from __future__ import print_function
import math
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from icp.icp import ICP


def PoseGraph(T_relative_LiDAR, robot_state_LiDAR, robot_xy, robot_truth,ICP_initializer, loop_length=10):
    '''
    Perform pose graph optimization using GTSAM with loop closure constraints.
    Args:
        T_relative_LiDAR (np.ndarray):  relative transforms between LiDAR poses
        robot_state_LiDAR (np.ndarray): estimated robot states from LiDAR ICP 
        robot_xy (np.ndarray):          LiDAR point cloud data in robo
        robot_truth (np.ndarray):       boolean mask for valid LiDAR points
        ICP_initializer (np.ndarray):   initial guess' for ICP 
        loop_length (int):              interval for adding loop closure constraints
    Returns:
        opt_traj(np.ndarray):           optimized robot states after pose graph optimization
    '''

    n_states = robot_state_LiDAR.shape[1]

    # Create noise models for the prior and odometry
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.1, 0.1, 0.1))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.1, 0.1, 0.1))

    # 1. Create a factor graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()

    # 2a. Add a prior on the first pose, setting it to the origin
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), PRIOR_NOISE))

    # 2b. Add odometry factors between consecutive poses
    for i in range(1,n_states):
        RelTransform = T_relative_LiDAR[i,:,:] # T_{i,i+1}
        del_x = RelTransform[0,-1]             # x translation
        del_y = RelTransform[1,-1]             # y translation
        del_p = np.array([[del_x],[del_y]])    # translation vector
        [[del_x],[del_y]] = np.matmul(-RelTransform[:2,:2].T,del_p)
        del_theta = np.arctan2(RelTransform[0,1],RelTransform[0,0]) # relative rotation angle
        graph.add(gtsam.BetweenFactorPose2(i, i+1, gtsam.Pose2(del_x, del_y, del_theta), ODOMETRY_NOISE))
    
    # 2b. Create the initial estimate for the poses using the LiDAR ICP estimated trajectory
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.0, 0.0, 0.0))

    for i in range(1,n_states):
        x = robot_state_LiDAR[0,i]
        y = robot_state_LiDAR[1,i]
        theta = robot_state_LiDAR[2,i]
        initial_estimate.insert(i+1, gtsam.Pose2(x, y, theta)) # add the initial estimate
    
    # 2c. Add loop closure constraint every loop_length steps
    for i in tqdm(range(n_states//loop_length-1)):
        i = int(loop_length*i+1)
        
        # calculate the initial guess for ICP
        ICP_guess = np.eye(3)
        for j in range(i,i+loop_length):
            ICP_guess = np.matmul(ICP_guess,ICP_initializer[j,:,:])
        ICP_guess = np.linalg.inv(ICP_guess)

        # get the point cloud data for ICP for pose i
        data1_x = robot_xy[0,:,i][robot_truth[:,i]]
        data1_y = robot_xy[1,:,i][robot_truth[:,i]]
        data1 = np.concatenate((data1_x.reshape((1,-1)),data1_y.reshape((1,-1))),axis=0)

        # get the point cloud data for ICP for pose i+loop_length
        datan_x = robot_xy[0,:,i+loop_length][robot_truth[:,i+loop_length]]
        datan_y = robot_xy[1,:,i+loop_length][robot_truth[:,i+loop_length]]
        datan = np.concatenate((datan_x.reshape((1,-1)),datan_y.reshape((1,-1))),axis=0)

        # run ICP b/w the two point clouds with the initial guess
        transform = ICP(datan,data1,ICP_guess,max_iter=200)
        transform.startIterations()
        T = transform.FinalTransformation

        # extract the relative transform from ICP result
        del_x = T[0,-1]
        del_y = T[1,-1]
        del_p = np.array([[del_x],[del_y]])
        [[del_x],[del_y]] = np.matmul(-T[:2,:2].T,del_p)
        del_theta = np.arctan2(T[0,1],T[0,0])

        # add the computed loop closure constraint (relative transform) to the graph
        graph.add(gtsam.BetweenFactorPose2(i+loop_length,i, gtsam.Pose2(del_x, del_y, del_theta), ODOMETRY_NOISE))

    # Set up and run the optimizer for the full graph
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # construct the optimised trajectory
    opt_traj = np.zeros((3,1))
    for i in range (1,n_states):
        x = result.atPose2(i).x()
        y = result.atPose2(i).y()
        theta = result.atPose2(i).theta()
        state = np.array([[x],[y],[theta]])
        opt_traj = np.hstack((opt_traj,state))

    # Plot the optimized trajectory
    plt.plot(opt_traj[0,:],opt_traj[1,:],label="GTSAM Optimised Trajectory")
    plt.legend()
    plt.show()

    return opt_traj