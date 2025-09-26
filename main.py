import numpy as np
from odometry import Odometry
from occupancy_texture import TextureMap, MatchImageDispState, OccupancyGrid
from pose_graph import PoseGraph
from icp.icp import LiDARICP, calcICPInitialGuess, LiDARDataHandler
import argparse


def main():
	#---------------------- Load all the data -------------------
	dataset = args.dataset
	with np.load("./data/Encoders%d.npz"%dataset) as data:
		encoder_counts = data["counts"]								# 4 x n encoder counts
		encoder_ts = data["time_stamps"] 							# encoder time stamps

	with np.load("./data/Hokuyo%d.npz"%dataset) as data:
		lidar_info = {key: data[key] for key in ['angle_min','angle_max','angle_increment','range_min','range_max']}
		lidar_ranges = data["ranges"]       						# range data [m] (Note: values < range_min or > range_max should be discarded)
		lidar_ts = data["time_stamps"]  							# acquisition times of the lidar scans
		
	with np.load("./data/Imu%d.npz"%dataset) as data:
		imu_angular_velocity = data["angular_velocity"] 			# angular velocity in rad/sec
		imu_linear_acceleration = data["linear_acceleration"] 		# accelerations in gs (gravity acceleration scaling)
		imu_data = np.vstack((imu_linear_acceleration,imu_angular_velocity))
		imu_ts = data["time_stamps"]  								# acquisition times of the imu measurements
	
	with np.load("./data/Kinect%d.npz"%dataset) as data:
		disp_ts = data["disparity_time_stamps"] 					# acquisition timestamps of the disparity images
		rgb_ts = data["rgb_time_stamps"] 							# acquisition timestamps of the rgb images


	#-------------- Get robot state from odometry data (IMU and encoder) ---------------
	Odom = Odometry(imu_data,imu_ts,encoder_counts,encoder_ts)
	Odom.PlotTrajectory(holdon=True)

	#--------------- Convert LiDAR (length,theta) scans to x,y coordinates in the robot frame ---------------
	LiDAR_data = LiDARDataHandler(lidar_ranges,lidar_info)
	robot_xy = LiDAR_data.lidar_xy
	robot_truth = LiDAR_data.lidar_truth

	# -------------- Get ICP Initial Guess from odometry measurements ---------------
	inital_icp_conditions = calcICPInitialGuess(Odom.all_states,Odom.time_stamps,robot_xy,lidar_ts)

	#--------------- Using ICP, get Transformations and Trajectory from Lidar scans ---------------
	print("LiDAR ICP based state estimation starting...")
	LiDAR_Transforms = LiDARICP(robot_xy,robot_truth,inital_icp_conditions)
	LiDAR_Transforms.PlotTrajectory(holdon=False)
	robot_state_LiDAR = LiDAR_Transforms.state			# estimated robot states from LiDAR ICP
	T_LiDAR = LiDAR_Transforms.all_rot_mat				# transforms from LiDAR poses to world frame
	T_relative_LiDAR = LiDAR_Transforms.T_relative		# relative transforms between LiDAR poses

	# -------------- Get the Occupancy Grid using states estimated from LiDAR ICP ---------------
	print("LiDAR ICP based occupancy grid starting...")
	OccupancyGrid(robot_xy, robot_truth, robot_state_LiDAR,title="Occupancy Grid - LiDAR ICP")

	# -------------- Match Image, Disparity and Lidar transforms for Texture Mapping ---------------
	ImageIndex, DispIndex, MatchedState = MatchImageDispState(rgb_ts,disp_ts,lidar_ts,robot_state_LiDAR,T_LiDAR)

	#--------------- Generate Floor Texture Map using LiDAR ICP based trajectory ---------------
	print("LiDAR ICP trajectory based texture map starting...")
	TextureMapLiDAR = TextureMap(MatchedState,DispIndex,ImageIndex,dataset,title="LiDAR ICP - Floor Texture Map")
	TextureMapLiDAR.generate()

	Odom.PlotTrajectory(holdon=True)
	LiDAR_Transforms.PlotTrajectory(holdon=True)

	#--------------- Implement pose graph optimization ---------------
	print("Pose Graph loop closure trajectory optimization starting...")
	pose_graph_opt_states = PoseGraph(T_relative_LiDAR,robot_state_LiDAR,robot_xy,robot_truth,inital_icp_conditions, loop_length=args.loop_length)

	#--------------- Get the Occupancy Grid using Optimized trajectory ---------------
	print("Optimized trajectory based occupancy grid starting...")
	OccupancyGrid(robot_xy, robot_truth, pose_graph_opt_states, title="Occupancy Grid - GTSAM")

	#--------------- Match Image, Disparity and Lidar transforms for Texture Mapping for pose graph Optimized states ---------------
	ImageIndex, DispIndex, PoseGraphMatchedState = MatchImageDispState(rgb_ts,disp_ts,lidar_ts,pose_graph_opt_states,T_LiDAR)

	#--------------- Generate Floor Texture Map using PG Optimized trajectory ---------------
	print("Optimized trajectory based texture map starting...")
	TextureMapPG = TextureMap(PoseGraphMatchedState,DispIndex,ImageIndex,dataset,title="Optimized Trajectory - Floor Texture Map")
	TextureMapPG.generate()
	print("exiting...")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=int, default=20, help='Dataset number (20 or 21)')
	parser.add_argument('--loop_length', type=int, default=10, help='Loop closure interval(default: 10)')
	args = parser.parse_args()

	main()
