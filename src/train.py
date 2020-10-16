"""
Deep Inverse Q-Learning with Constraints. NeurIPS 2020.
Gabriel Kalweit, Maria Huegle, Moritz Werling and Joschka Boedecker
Neurorobotics Lab, University of Freiburg

This script collects data from the Objectworld environment. Run with python collect_data.py n_traj1, n_traj2 ... , where n_traj1, n_traj2 are possible numbers of trajectories. To collect 4,32 and 128 and 512 trajectories use:

python train.py alg n_traj

"""

import numpy as np
np.set_printoptions(suppress=True)
import sys, os
import time
import random
import string
import time
import pickle

from mdp.objectworld import Objectworld 
from algorithms.iavi import inverse_action_value_iteration
from algorithms.iql import inverse_q_learning


              
if __name__ == "__main__":
    alg = sys.argv[1]
    supported_algorithms = [ "iavi", "iql"]
    assert(alg in supported_algorithms)

    n_traj = int(sys.argv[2])

    if alg == "iql":
        updates_or_epochs = int(sys.argv[3])
    else:
        updates_or_epochs = 1

    data_file = "objectworld_%s_trajectories"%n_traj
    data_dir = os.path.join("../data", data_file)

    sets = sorted(next(os.walk(data_dir))[1])
    for current_set in sets:
        print(current_set)
        set_dir = os.path.join(data_dir, current_set)
     
        store_dir = os.path.join("../results", data_file, current_set, "%s_%s"%(alg, updates_or_epochs))
        os.makedirs(store_dir, exist_ok=True)

        gamma = 0.9
        n_actions = 5

        feature_matrix = np.load(os.path.join(set_dir, "feature_matrix.npy"))
        trajectories = np.load(os.path.join(set_dir, "trajectories.npy"))
        action_probabilities = np.load(os.path.join(set_dir, "action_probabilities.npy"))
        transition_probabilities = np.load(os.path.join(set_dir, "transition_probabilities.npy"))
        ground_r= np.load(os.path.join(set_dir, "ground_r.npy"))
        p_start_state = np.load(os.path.join(set_dir, "p_start_state.npy"))

        is_terminal = np.zeros((feature_matrix.shape[0], n_actions))

        start = time.time()
       
        if alg == "iavi":
            q, r, boltz = inverse_action_value_iteration(feature_matrix, n_actions, gamma, transition_probabilities, action_probabilities, theta=0.01)
        elif alg == "iql":
            q, r, boltz = inverse_q_learning(feature_matrix, n_actions,  gamma, trajectories, \
                                         alpha_r=0.0001, alpha_q=0.01, alpha_sh=0.01, epochs=updates_or_epochs, real_distribution=action_probabilities)
        else:
            print("Algorithm not supported.")

        end = time.time()

        np.save(os.path.join(store_dir, "runtime"), (end - start))
        np.save(os.path.join(store_dir, "r"), r)
        np.save(os.path.join(store_dir, "q"), q)
        np.save(os.path.join(store_dir, "boltzmann"), boltz)
