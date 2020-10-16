"""
Deep Inverse Q-Learning with Constraints. NeurIPS 2020.
Gabriel Kalweit, Maria Huegle, Moritz Werling and Joschka Boedecker
Neurorobotics Lab, University of Freiburg.
"""

import numpy as np
epsilon = 1e-6


def inverse_q_learning(feature_matrix,nA, gamma, transitions, alpha_r, alpha_q, alpha_sh, epochs, real_distribution):
    """
    Implementation of IQL from Deep Inverse Q-learning with Constraints. Gabriel Kalweit, Maria Huegle, Moritz Wehrling and Joschka Boedecker. NeurIPS 2020.
    Arxiv : https://arxiv.org/abs/2008.01712
    """
    nS = feature_matrix.shape[0]

    
    # initialize tables for reward function, value functions and state-action visitation counter.
    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))
    q_sh = np.zeros((nS, nA))
    state_action_visitation = np.zeros((nS, nA))

    for i in range(epochs):
        if i%10 == 0:
            print("Epoch %s/%s" %(i+1, epochs))
       
        for traj in transitions:
            for (s, a, _, ns) in traj:
                state_action_visitation[s][a] += 1
                d = False   # no terminal state

                # compute shifted q-function.
                q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + alpha_sh * (gamma * (1-d) * np.max(q[ns]))
                
                # compute log probabilities.
                sum_of_state_visitations = np.sum(state_action_visitation[s])
                log_prob = np.log((state_action_visitation[s]/sum_of_state_visitations) + epsilon)
                
                # compute eta_a and eta_b for Eq. (9).
                eta_a = log_prob[a] - q_sh[s][a]
                other_actions = [oa for oa in range(nA) if oa != a]
                eta_b = log_prob[other_actions] - q_sh[s][other_actions]
                sum_oa = (1/(nA-1)) * np.sum(r[s][other_actions] - eta_b)

                # update reward-function.
                r[s][a] = (1-alpha_r) * r[s][a] + alpha_r * (eta_a + sum_oa)

                # update value-function.
                q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * (r[s, a] + gamma * (1-d) * np.max(q[ns]))
                s = ns

    # compute Boltzmann distribution.
    boltzman_distribution = []
    for s in range(nS):
        boltzman_distribution.append([])
        for a in range(nA):
            boltzman_distribution[-1].append(np.exp(q[s][a]))
    boltzman_distribution = np.array(boltzman_distribution)
    boltzman_distribution /= np.sum(boltzman_distribution, axis=1).reshape(-1, 1)
    return q, r, boltzman_distribution
