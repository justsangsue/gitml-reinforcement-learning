from policy_value import *
from mdp import *
import time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    best_value = [0.0, 0.64, 0.8, 1.0, 0.8, 0.64, 0.0, 0.0, 0.0]
    
    """Policy Iteration"""
    mdp = Mdp()
    policy_value = Policy_Value(mdp)
    error_policy_iter = []
    x_p = []
    run_time_p = []
    accumu_time = 0.0
    for i in range(100):
        if i == 1:
            print policy_value.pi
        start = time.time()
        x_p.append(i)
        policy_value.policy_evaluate()
        policy_value.policy_improvement()
        error_policy_iter.append(pow((sum(best_value) - sum(policy_value.v)), 2))
        end = time.time()
        accumu_time += (end - start) * 1000.0
        run_time_p.append(accumu_time)
    print policy_value.pi

    """Value Iteration"""
    mdp = Mdp()
    policy_value = Policy_Value(mdp)
    error_value_iter = []
    x_v = []
    run_time_v = []
    accumu_time = 0.0
    for i in range(100):
        if i == 1:
            print policy_value.pi
        start = time.time()
        x_v.append(i)
        policy_value.value_iteration()
        error_value_iter.append(pow((sum(best_value) - sum(policy_value.v)), 2))
        end = time.time()
        accumu_time += (end - start) * 1000.0
        run_time_v.append(accumu_time)
    print policy_value.pi

    plt.plot(x_p, error_policy_iter, "-.", label="policy iteration")
    plt.plot(x_v, error_value_iter, "-", label="value iteration")
    plt.xticks(np.arange(0, 100, 10))
    plt.xlabel("Iteration Number")
    plt.ylabel("Squared Error")
    plt.grid(True)
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.legend()
    plt.show()

    plt.plot(x_p, run_time_p, "-.", label="policy iteration")
    plt.plot(x_v, run_time_v, "-", label="value iteration")
    plt.xticks(np.arange(0, 100, 10))
    plt.xlabel("Iteration Number")
    plt.ylabel("Running Time")
    plt.grid(True)
    plt.grid(color='silver', linestyle='--', linewidth=.5)
    plt.legend()
    plt.show()
