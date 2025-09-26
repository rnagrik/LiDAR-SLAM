import numpy as np
from utils.pr2_utils import bresenham2D
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import os

def OccupancyGrid(robot_xy, robot_truth, robot_state, grid_res=0.05, xlims=(-30,30), ylims=(-30,30), title=None):
    '''
    Create an occupancy grid map from the robot poses and LiDAR scans.
    args:
        robot_xy(np.ndarray):     2 x N x M array of LiDAR points in the robot frame (N points per scan, M scans)
        robot_truth(np.ndarray):  N x M boolean array indicating valid LiDAR points
        robot_state(np.ndarray):  3 x M array of robot poses (x, y, theta) for each scan
        grid_res(float):          resolution of the occupancy grid in meters
        xlims(tuple):             (xmin, xmax) grid limits (in m)
        ylims(tuple):             (ymin, ymax) grid limits (in m)
        title(str):               optional title for the occupancy grid plot
    '''
  
    instances = robot_xy.shape[2] 

    # init MAP structure
    MAP = {}
    MAP['res']   = grid_res  
    MAP['xmin']  = xlims[0]  
    MAP['ymin']  = ylims[0]  
    MAP['xmax']  =  xlims[1]
    MAP['ymax']  =  ylims[1]
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

    print("Building Occupancy Grid")

    # for each LiDAR scan, transform the points from robot frame to world frame and update the occupancy grid
    for i in tqdm(range(instances)):
        # get the robot pose
        x_r,y_r,theta_r = robot_state[0,i],robot_state[1,i],robot_state[2,i]
        # get the transformation matrix from robot to world frame
        T = np.array([[np.cos(theta_r),-np.sin(theta_r),x_r],[np.sin(theta_r),np.cos(theta_r),y_r],[0,0,1]])
        # transform the LiDAR points to world frame
        x = robot_xy[0,:,i][robot_truth[:,i]]
        y = robot_xy[1,:,i][robot_truth[:,i]]
        n_points = x.shape[0] # number of valid points
        # homogenize the points
        h = np.ones(n_points)
        p_r = np.array([[x],[y],[h]]).reshape(3,h.shape[0])  # LiDAR points in robot frame
        p_w = np.matmul(T,p_r)                               # LiDAR points in world frame
        # get the grid cell indices for the valid points in the world frame
        ei = np.ceil((p_w[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        ej = np.ceil((p_w[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        # get the grid cell indices for the current robot position in the world frame
        start_i = np.ceil((robot_state[0,i] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        start_j = np.ceil((robot_state[1,i] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1

        # for each valid point, trace the ray from the robot to the point and update the occupancy grid
        for point_index in range(n_points):
            # get the end point of the ray
            end_i, end_j = ei[point_index], ej[point_index]
            # get the points along the ray using Bresenham's line algorithm
            bresenham_points = bresenham2D(start_i, start_j, end_i, end_j)[:,:-1].astype(int)  # removed the last point since it is the occupied point
            free_points_x,free_points_y = bresenham_points[0,:],bresenham_points[1,:]
            # get the valid indices for the free points (i.e. inside the map and not occupied)
            indGood = np.logical_and(np.logical_and(np.logical_and((free_points_x > 1), (free_points_y > 1)), (free_points_x < MAP['sizex']-1)), (free_points_y < MAP['sizey']-1))
            free_points_x,free_points_y = free_points_x[indGood],free_points_y[indGood]
            # cap the maximum value of the occupancy grid to avoid overflow
            cap = 30.0
            free_points_x = free_points_x[MAP['map'][bresenham_points[0,:][indGood],bresenham_points[1,:][indGood]]<cap]
            free_points_y = free_points_y[MAP['map'][bresenham_points[0,:][indGood],bresenham_points[1,:][indGood]]<cap]
            # update the occupancy grid for end point, if it is inside the map and not occupied
            if np.logical_and(np.logical_and(np.logical_and((end_i > 1), (end_j > 1)), (end_i < MAP['sizex'])), (end_j < MAP['sizey'])):
                if -1.0*cap < MAP['map'][end_i,end_j] <cap:
                    MAP['map'][end_i,end_j] -= 1 # end point object 
            # update the occupancy grid for free points
            MAP['map'][free_points_x,free_points_y] += 1

    # calculate the robot trajectory (in the grid frame)
    traj_ij = np.zeros((2,1))
    for i in range(instances):
        start_i = np.ceil((robot_state[0,i] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        start_j = np.ceil((robot_state[1,i] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        traj_ij = np.hstack((traj_ij,np.array([[start_i],[start_j]])))
    traj_ij = traj_ij[:,1:]

    # get the occupancy probability from the log-odds values
    MAP['map'] = np.exp(MAP['map'], dtype=np.float64)/(1+np.exp(MAP['map'], dtype=np.float64))

    # plot the occupancy grid
    plt.figure()    
    plt.imshow(np.rot90(MAP['map']),cmap="gist_earth")  
    plt.plot(traj_ij[0,:],MAP['map'].shape[0] - traj_ij[1,:],c='red',label="Robot Trajectory")
    if str(title) != 'None':
        plt.title(title)
    colorbar = plt.colorbar()
    colorbar.set_ticks([0,0.5,1])
    colorbar.set_ticklabels(['Occupied', 'Unexplored', 'Free Space'])
    plt.legend()
    plt.show()



def MatchImageDispState(image_ts, disp_ts, state_ts, robot_state, transformations):
    '''
    Match the image timestamps with the closest disparity and state timestamps.
    args:
        image_ts(np.ndarray):        1 x N array of image timestamps
        disp_ts(np.ndarray):         1 x M array of disparity timestamps
        state_ts(np.ndarray):        1 x K array of state timestamps
        robot_state(np.ndarray):     3 x K array of robot states (x, y, theta) for each state timestamp
        transformations(np.ndarray): K x 3 x 3 array of transformation matrices for each state timestamp
    '''
    ImageIndex = np.arange(image_ts.shape[0])
    MatchedDispIndex = np.zeros(image_ts.shape[0])
    MatchedState = np.zeros((3,1))
    # MatchedTransforms = np.zeros((1,3,3))

    for image_index in ImageIndex:
        # find the closest state timestamp to the image timestamp
        diff_state_ts = np.abs(state_ts - image_ts[image_index])
        index_state = diff_state_ts.argmin()
        # self.MatchedTransforms = np.concatenate((self.MatchedTransforms,self.__transformations[index_state,:,:].reshape((1,3,3))),axis=0)
        # stack the matched state
        MatchedState = np.hstack((MatchedState,robot_state[:,index_state].reshape((3,1))))
        # find the closest disparity timestamp to the image timestamp
        diff_disp_ts = np.abs(disp_ts - image_ts[image_index])
        index_disp = diff_disp_ts.argmin()
        MatchedDispIndex[image_index] = index_disp
    
    # self.MatchedTransforms = self.MatchedTransforms[1:,:,:]
    MatchedState = MatchedState[:,1:]

    return ImageIndex, MatchedDispIndex, MatchedState


class TextureMap:
    '''
    Create a texture map of the floor using the robot states and the RGBD images.
    args:
        robot_state(np.ndarray):     3 x N array of robot states (x, y, theta) for every image
        disp_idxs(np.ndarray):       1 x N array of disparity indices for each image
        image_idxs(np.ndarray):      1 x N array of image indices
        dataset_number(int):         dataset number (to load the images)
        title(str):                  optional title for the texture map plot
        make_gif(bool):              whether to save each step as a png image for making a gif later
    '''

    def __init__(self, robot_state, disp_idxs, image_idxs, dataset_number, title="", make_gif=False):
        self.dataset_number = dataset_number
        self.image_index_list = image_idxs
        self.disp_index_list = disp_idxs
        self.title = title
        self.make_gif = make_gif
        self.robot_state = robot_state

        if self.make_gif and not os.path.exists(f'./images/{dataset_number}/{self.title}'):
            os.makedirs(f'./images/{dataset_number}/{self.title}')

    def normalize(img):
        max_, min_ = img.max(), img.min()
        return (img - min_)/(max_-min_)
    
    def __GetRobotXYZColour(self, image_index, disp_index):
        '''
        Get the 3D coordinates of the points in the camera frame and the corresponding RGB values.
        args:
            image_index(int):   index of the RGB image
            disp_index(int):    index of the disparity image
        returns:
            x_c(np.ndarray):    x coordinates of the points in the camera frame
            y_c(np.ndarray):    y coordinates of the points in the camera frame
            z_c(np.ndarray):    z coordinates of the points in the camera frame
            imc(np.ndarray):    RGB image
            rgbu(np.ndarray):   u coordinates of the points in the RGB image
            rgbv(np.ndarray):   v coordinates of the points in the RGB image
        '''

        disp_path = f"./data/dataRGBD/Disparity{self.dataset_number}/"
        rgb_path = f"./data/dataRGBD/RGB{self.dataset_number}/"
        
        # load RGBD images
        imd = cv2.imread(disp_path+f'disparity{self.dataset_number}_{disp_index}.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
        imc = cv2.imread(rgb_path+f'rgb{self.dataset_number}_{image_index}.png')[...,::-1] # (480 x 640 x 3)
    
        # convert from disparity from uint16 to double
        disparity = imd.astype(np.float32)
    
        # get depth
        dd = (-0.00304 * disparity + 3.31)
        z = 1.03 / dd
    
        # calculate u and v coordinates 
        v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    
        # get 3D coordinates 
        fx,fy = 585.05108211,585.05108211
        cx,cy = 315.83800193,242.94140713
        x = (u-cx) / fx * z
        y = (v-cy) / fy * z
    
        # calculate the location of each pixel in the RGB image
        rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
        rgbv = np.round((v * 526.37 + 16662.0)/fy)

        # get the valid indices (i.e. inside the RGB image and with valid depth)
        valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

        # transform the points to the robot frame and get the valid points
        x_c,y_c,z_c = z,-x,-y
        x_c,y_c,z_c,imc,rgbu,rgbv = x_c[valid],y_c[valid],z_c[valid],imc,rgbu[valid].astype(int),rgbv[valid].astype(int)

        return x_c,y_c,z_c,imc,rgbu,rgbv

    def generate(self,res=0.05,xlims=(-30,30),ylims=(-30,30)):
        '''
        Generate a texture map of the floor using the robot states and the RGBD images.
        args:
            res(float):          resolution of the texture map in meters
            xlims(tuple):        (xmin, xmax) grid limits (in m)
            ylims(tuple):        (ymin, ymax) grid limits (in m)
        '''

        # init MAP structure
        MAP = {}
        MAP['res']   = res 
        MAP['xmin']  = xlims[0]  
        MAP['ymin']  = ylims[0]
        MAP['xmax']  =  xlims[1]
        MAP['ymax']  =  ylims[1]
        MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map']   = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=np.uint8)

        # calculate the transformation from camera frame to robot frame
        CamToRobotTransform = np.eye(4)
        p = np.array([0.18,0.005,0.36])  # known translation from camera to robot frame
        pitch, yaw = 0.36, 0.021         # known rotation from camera to robot frame
        R = np.matmul(np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]]),np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]))
        CamToRobotTransform[:3,:3] = R
        CamToRobotTransform[:3,-1] = p

        plt.figure()
        print("Building Texture Map...")

        # for each image, transform the points from camera frame to world frame and update the map
        for image_index in tqdm(self.image_index_list):
            # get the corresponding disparity index
            disp_index = self.disp_index_list[image_index]
            # get the robot pose at the time of image capture
            x_r,y_r,theta_r = self.robot_state[0,image_index],self.robot_state[1,image_index],self.robot_state[2,image_index]
            transformation = np.array([[np.cos(theta_r),-np.sin(theta_r),x_r],[np.sin(theta_r),np.cos(theta_r),y_r],[0,0,1]])
            # get the transformation matrix from robot to world frame
            RobToWorldTransform = np.eye(4)
            RobToWorldTransform[:2,:2] = transformation[:2,:2]
            RobToWorldTransform[:2,-1] = transformation[:2,-1]
            # get the 3D points in camera frame and the corresponding RGB values
            x_c,y_c,z_c,imc,rgbu,rgbv = self.__GetRobotXYZColour(int(image_index+1),int(disp_index+1))
            # transform points from camera frame to world frame
            points_camera_frame = np.array([x_c,y_c,z_c,np.ones(x_c.shape[0])]).reshape(4,x_c.shape[0])
            points_robot_frame = np.matmul(CamToRobotTransform,points_camera_frame)
            points_world_frame = np.matmul(RobToWorldTransform,points_robot_frame)
            [x_w,y_w,z_w,_] = points_world_frame
            # eliminate invalid points (i.e points below the ground)
            x_w = x_w[z_w<=0]
            y_w = y_w[z_w<=0]
            rgbu = rgbu[z_w<=0]
            rgbv = rgbv[z_w<=0]
            # get the grid cell indices for the valid points in the world frame
            point_i = np.ceil((x_w - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
            point_j = np.ceil((y_w - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
            # get the valid indices for the points (i.e. inside the map) and update the texture map
            indGood = np.logical_and(np.logical_and(np.logical_and((point_i > 1), (point_j > 1)), (point_i < MAP['sizex'])), (point_j < MAP['sizey']))
            MAP['map'][point_i[indGood],point_j[indGood],:] = imc[rgbv[indGood],rgbu[indGood]]
            if self.make_gif and image_index % 10 == 0:
                plt.imsave(f'./images/{self.dataset_number}/{self.title}/{image_index:04d}.png',np.rot90(MAP['map']), cmap="gray",vmin=-1,vmax=1)

        # plot the texture map
        plt.imshow(np.rot90(MAP['map']),cmap="gray",vmin=-1,vmax=1)
        if str(self.title) != 'None':
            plt.title(self.title)
        plt.show()
