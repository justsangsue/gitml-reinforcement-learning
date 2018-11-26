from mdp import *
from random import random
import matplotlib.pyplot as plt
import numpy as np
import time


def epsilon_greedy(qfunc, state, actions, epsilon):
    """Find the action that gives highest q value"""
    amax = 0
    key = "%d_%s"%(state, actions[0])
    #print qfunc
    qmax = qfunc[key]
    for i in range(len(actions)):
        key = "%d_%s"%(state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i

    """Initilize probability distribution"""
    prob = [0.0 for i in range(len(actions))]
    prob[amax] += 1 - epsilon
    for i in range(len(actions)):
        prob[i] += epsilon / len(actions)

    """Choose action based on probability"""
    r = random()
    s = 0.0
    for i in range(len(actions)):
        s += prob[i]
        if s >= r:
            return actions[i]
    return actions[len(actions) - 1]


"""SARSA algorithm"""
def sarsa(num_iter, alpha, epsilon, qbest):
    """
    num_iter: number of iterations
    alpha: learning rate
    epsilon: greedy index
    """
    x = []
    y = []
    run_time = []
    accumu_time = 0.0

    """Initialize Q values for all possible (s, a) combinations"""
    gamma = 0.8
    mdp = Mdp()
    states = mdp.states
    actions = mdp.actions
    qfunc = dict()
    for state in states:
        for action in actions:
            key = "%d_%s"%(state, action)
            qfunc[key] = 0.0

    """Update q values"""
    for i in range(num_iter):
        if i == 1:
            print qfunc
        start = time.time()
        x.append(i)
        y.append(compute_error(qfunc, qbest))
        state = states[int(random() * len(states))]
        action = actions[int(random() * len(actions))]
        t = False
        counter = 0
        while not t and counter < 100:
            key = "%d_%s"%(state, action)
            t, s, r = mdp.transform(state, action)
            a = epsilon_greedy(qfunc, s, actions, epsilon)
            new_key = "%d_%s"%(s, a)
            qfunc[key] = qfunc[key] + alpha * (\
                r + gamma * qfunc[new_key] - qfunc[key])
            state = s
            action = a
            counter += 1
        end = time.time()
        accumu_time += (end - start) * 1000.0
        run_time.append(accumu_time)
    
    print qfunc
    return x, y, qfunc, run_time

def qlearning(num_iter, alpha, epsilon, qbest):
    """
    num_iter: number of iterations
    alpha: learning rate
    epsilon: greedy index
    """
    x = []
    y = []
    run_time = []
    accumu_time = 0.0

    """Initialize Q values for all possible (s, a) combinations"""
    gamma = 0.8
    mdp = Mdp()
    states = mdp.states
    actions = mdp.actions
    qfunc = dict()
    for state in states:
        for action in actions:
            key = "%d_%s"%(state, action)
            qfunc[key] = 0.0

    """Update q values"""
    for i in range(num_iter):
        if i == 1:
            print qfunc
        start = time.time()       
        x.append(i)
        y.append(compute_error(qfunc, qbest))
        state = states[int(random() * len(states))]
        action = actions[int(random() * len(actions))]
        t = False
        counter = 0
        while not t and counter < 100:
            key = "%d_%s"%(state, action)
            t, s, r = mdp.transform(state, action)
            qmax = -1.0
            for action in actions:
                if qmax < qfunc["%d_%s"%(s, action)]:
                    qmax = qfunc["%d_%s"%(s, action)]
            qfunc[key] = qfunc[key] + alpha * (\
                r + gamma * qmax - qfunc[key])
            state = s
            action = epsilon_greedy(qfunc, s, actions, epsilon)
            counter += 1
        end = time.time()
        accumu_time += (end - start) * 1000.0
        run_time.append(accumu_time)
    
    print qfunc
    return x, y, qfunc, run_time

def test_mode_free():
    print sarsa(100, 0.5, 0.1)
    print qlearning(100, 0.5, 0.1)

def compute_error(qfunc, qbest):
    se = 0.0
    for key in qfunc:
        error = qfunc[key] - qbest[key]
        se += error * error
    return se

if __name__ == "__main__":
    #test_mode_free()
    qbest = dict();
    with open("best_qfunc") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:  continue
            eles = line.split(":")
            qbest[eles[0]] = float(eles[1])
    
    """Plot SARSA and Q-learning"""
    iter_num = 1000
    alphas = [0.1, 0.5, 0.9]
    epsilons = [0.1, 0.5, 0.9]
    for alpha in alphas:
        for epsilon in epsilons:
            s_x, s_y, s_qfunc, s_time = sarsa(iter_num, alpha, epsilon, qbest)
            q_x, q_y, q_qfunc, q_time = qlearning(iter_num, alpha, epsilon, qbest)
            plt.plot(s_x, s_y, "-.", label="sarsa alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
            plt.plot(q_x, q_y, "-", label="qlearning alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
            plt.xticks(np.arange(0, 1000, 100))
            plt.xlabel("Iteration Number")
            plt.ylabel("Squared Error")
            plt.grid(True)
            plt.grid(color='silver', linestyle='--', linewidth=.5)
            plt.legend()
            plt.show()

            plt.plot(s_x, s_time, "-.", label="sarsa alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
            plt.plot(q_x, q_time, "-", label="qlearning alpha=%2.1f epsilon=%2.1f"%(alpha,epsilon))
            plt.xticks(np.arange(0, 1000, 100))
            plt.xlabel("Iteration Number")
            plt.ylabel("Running Time")
            plt.grid(True)
            plt.grid(color='silver', linestyle='--', linewidth=.5)
            plt.legend()
            plt.show()

    









            

