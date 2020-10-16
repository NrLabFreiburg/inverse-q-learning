"""
Deep Inverse Q-Learning with Constraints. NeurIPS 2020.
Gabriel Kalweit, Maria Huegle, Moritz Werling and Joschka Boedecker
Neurorobotics Lab, University of Freiburg.
"""

import numpy as np
epsilon = 1e-6


def inverse_action_value_iteration(feature_matrix, nA, gamma, transition_probabilities, action_probabilities, theta=0.0001):
    """
    Implementation of IAVI from Deep Inverse Q-learning with Constraints. Gabriel Kalweit, Maria Huegle, Moritz Wehrling and Joschka Boedecker. NeurIPS 2020.
    Arxiv : https://arxiv.org/abs/2008.01712
    """
    nS = feature_matrix.shape[0]

    # initialize tables for reward function and value function.
    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))

    # compute reverse topological order.
    T = []
    for i in reversed(range(nS)):
      T.append([i])

    # do while change in r over iterations is larger than theta.
    diff = np.inf
    while diff > theta:
        print(diff)
        diff = 0
        for t in T[0:]:
            for i in t:
                # compute coefficient matrix X_A(s) as in Eq. (9).
                X = []
                for a in range(nA):
                    row = np.ones(nA)
                    for oa in range(nA):
                        if oa == a:
                            continue
                        row[oa] /= -(nA-1)
                    X.append(row)
                X = np.array(X)

                # compute target vector Y_A(s) as in Eq. (9).
                y = []
                for a in range(nA):
                    other_actions = [oa for oa in range(nA) if oa != a]
                    sum_of_oa_logs = np.sum([np.log(action_probabilities[i][oa] + epsilon) for oa in other_actions])
                    sum_of_oa_q = np.sum([transition_probabilities[i][oa] * gamma * np.max(q[np.arange(nS)], axis=1) for oa in other_actions])
                    y.append(np.log(action_probabilities[i][a] + epsilon)-(1/(nA-1))*sum_of_oa_logs+(1/(nA-1))*sum_of_oa_q-np.sum(transition_probabilities[i][a] * gamma * np.max(q[np.arange(nS)], axis=1)))
                y = np.array(y)

                # Find least-squares solution.
                x = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                for a in range(nA):
                    diff = max(np.abs(r[i, a]-x[a]), diff)

                # compute new r and Q-values.
                r[i] = x
                for a in range(nA):
                    q[i, a] = r[i, a] + np.sum(transition_probabilities[i][a] * gamma * np.max(q[np.arange(nS)], axis=1))
    
    # calculate Boltzmann distribution.
    boltzman_distribution = []
    for s in range(nS):
        boltzman_distribution.append([])
        for a in range(nA):
            boltzman_distribution[-1].append(np.exp(q[s][a]))
    boltzman_distribution = np.array(boltzman_distribution)
    boltzman_distribution /= np.sum(boltzman_distribution, axis=1).reshape(-1, 1)
    return q, r, boltzman_distribution
