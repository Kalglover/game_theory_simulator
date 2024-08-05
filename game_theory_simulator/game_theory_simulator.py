import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def leader_cost(p, q, a, b):
    return a * p ** 2 - b * np.log(1 + p / q)

def follower_cost(q, p, c, d):
    return c * q ** 2 - d * np.log(1 + q / p)

def follower_response(p, c, d, initial_guess=0.1):
    result = minimize(lambda q: follower_cost(q, p, c, d), initial_guess, bounds=[(0.01, None)])
    return result.x[0]

def find_stackelberg_equilibrium(a, b, c, d):
    def objective(p):
        q = follower_response(p, c, d)
        return leader_cost(p, q, a, b)

    initial_guess = 0.1
    bounds = [(0.01, None)]
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_p = result.x[0]
    optimal_q = follower_response(optimal_p, c, d)
    return optimal_p, optimal_q

def plot_results(p, q, powers, responses):
    plt.figure(figsize=(10, 6))
    plt.plot(powers, responses, label='Follower response')
    plt.scatter([p], [q], color='red', label='Equilibrium Point')
    plt.title('Follower Response vs Leader Power Level')
    plt.xlabel('Leader Power Level (p)')
    plt.ylabel('Follower Power Level (q)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    a, b, c, d = 1, 2, 1, 2
    p_star, q_star = find_stackelberg_equilibrium(a, b, c, d)

    powers = np.linspace(0.01, 3, 100)
    responses = [follower_response(p, c, d) for p in powers]

    plot_results(p_star, q_star, powers, responses)

if __name__ == "__main__":
    main()
