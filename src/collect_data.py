"""
Deep Inverse Q-Learning with Constraints. NeurIPS 2020.
Gabriel Kalweit, Maria Huegle, Moritz Werling and Joschka Boedecker
Neurorobotics Lab, University of Freiburg

This script collects data from the Objectworld environment. Run with python collect_data.py n_traj1, n_traj2 ... , where n_traj1, n_traj2 are possible numbers of trajectories. To collect 4,32 and 128 and 512 trajectories use:

python collect_data.py 4 32 128 512

"""

import sys, os
import numpy as np
np.set_printoptions(suppress=True)
from mdp.objectworld import Objectworld 
              

if __name__ == "__main__":

    num_sets = 1        # number of runs for each experiment.
    n_traj = [int(n) for n in sys.argv[1:]]

    # Objectworld settings.
    grid_size = 32
    n_objects = 50
    n_colours = 2
    wind = 0.3
    discount = 0.99
    trajectory_length = 8

    env = Objectworld(grid_size, n_objects, n_colours, wind, discount)
    feature_matrix = env.feature_matrix(discrete=False)


    for n in n_traj:
        print("%s trajectories from [%s]"%(n, n_traj))
        for i in range(num_sets):
            print("\tset %s/%s"%(i+1, num_sets))
            store_dir = os.path.join("../data", "objectworld_%s_trajectories"%n, "objectworld_%s_trajectories_set_%s"%(n, i))
            os.makedirs(store_dir)

            n_trajectories = n
            trajectories, action_probabilities, transition_probabilities, ground_r= env.collect_demonstrations(n_trajectories, trajectory_length)
            np.save(os.path.join(store_dir, "trajectories.npy"), trajectories)
            np.save(os.path.join(store_dir, "action_probabilities.npy"), action_probabilities)
            np.save(os.path.join(store_dir, "transition_probabilities.npy"), transition_probabilities)
            np.save(os.path.join(store_dir, "ground_r.npy"), ground_r)

            p_start_state = (np.bincount(trajectories[:, 0, 0], minlength=env.n_states)/trajectories.shape[0])
            np.save(os.path.join(store_dir, "p_start_state.npy"), p_start_state)
            np.save(os.path.join(store_dir, "feature_matrix.npy"), env.feature_matrix(discrete=False))
