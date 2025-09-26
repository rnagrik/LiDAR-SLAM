import numpy as np
from icp.utils_icp import read_canonical_model, load_pc, visualize_icp_result
from icp.icp import ICP
from scipy.stats import ortho_group
from scipy.spatial.transform import Rotation
import argparse


def main():

    # Load canonical model and point clouds
    source_pc = read_canonical_model(args.object)
    num_pc = 4 # number of point clouds

    for i in range(num_pc):
        print(f"Performing ICP on {args.object} point cloud ({i+1}/{num_pc})")
        target_pc = load_pc(args.object, i)    

        T_guess = np.eye(4)                          # initial guess for ICP
        if args.object == 'drill' and i in [2,3]:    # better initial guess for this case
            T_guess[:2,:2] = -T_guess[:2,:2]
        T_guess[:-1,-1] = np.mean(target_pc.T,axis=1)-np.mean(source_pc.T,axis=1)

        icp = ICP(source_pc.T,target_pc.T,T_guess,max_iter=args.max_iter)
        icp.startIterations()

        T_final = icp.FinalTransformation
        visualize_icp_result(source_pc, target_pc, T_final)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object',   type=str, default='drill', help='object name: drill or liq_container')
    parser.add_argument('--max_iter', type=int, default=200,     help='maximum number of ICP iterations')
    args = parser.parse_args()

    main()