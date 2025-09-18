# ================================
# Infinite horizon policy evaluation
# ================================

# Loading modules
import numpy as np  # array operations

# Infinite horizon policy evaluation function using value iteration
def evaluate_pi_infinite(pi, P, r, gamma, max_iterations=1000, tolerance=1e-6):
    """
    Evaluate policy for infinite horizon MDP using value iteration
    
    Inputs:
    pi: policy matrix (S x A) - deterministic policy
    P: transition probabilities (S x S x A)
    r: rewards (S x A)
    gamma: discount factor
    max_iterations: maximum number of iterations
    tolerance: convergence tolerance
    
    Outputs:
    V_pi: value function (S,)
    """
    
    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions
    
    # Initialize value function
    V_pi = np.zeros(S)
    
    # Value iteration
    for iteration in range(max_iterations):
        V_pi_old = V_pi.copy()
        
        for s in range(S):
            a = int(pi[s])  # action taken in state s
            V_pi[s] = r[s, a] + gamma * np.dot(P[s, :, a], V_pi_old)
        
        # Check for convergence
        if np.max(np.abs(V_pi - V_pi_old)) < tolerance:
            break
    
    return V_pi

# Infinite horizon policy evaluation in terms of events function
def evaluate_events_infinite(pi, P, event_states, gamma, max_iterations=1000, tolerance=1e-6):
    """
    Evaluate expected number of events for infinite horizon MDP
    
    Inputs:
    pi: policy matrix (S x A) - deterministic policy
    P: transition probabilities (S x S x A)
    event_states: event indicators (S,)
    gamma: discount factor
    max_iterations: maximum number of iterations
    tolerance: convergence tolerance
    
    Outputs:
    E_pi: expected number of events (S,)
    """
    
    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions
    
    # Initialize expected events
    E_pi = np.zeros(S)
    
    # Value iteration for events
    for iteration in range(max_iterations):
        E_pi_old = E_pi.copy()
        
        for s in range(S):
            a = int(pi[s])  # action taken in state s
            E_pi[s] = event_states[s] + gamma * np.dot(P[s, :, a], E_pi_old)
        
        # Check for convergence
        if np.max(np.abs(E_pi - E_pi_old)) < tolerance:
            break
    
    return E_pi

# Policy improvement for infinite horizon MDP
def policy_improvement_infinite(P, r, gamma, max_iterations=1000, tolerance=1e-6):
    """
    Policy improvement algorithm for infinite horizon MDP
    
    Inputs:
    P: transition probabilities (S x S x A)
    r: rewards (S x A)
    gamma: discount factor
    max_iterations: maximum number of iterations
    tolerance: convergence tolerance
    
    Outputs:
    pi_opt: optimal policy (S,)
    V_opt: optimal value function (S,)
    """
    
    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions
    
    # Initialize random policy
    pi_opt = np.random.randint(0, A, S)
    
    # Policy iteration
    for iteration in range(max_iterations):
        pi_old = pi_opt.copy()
        
        # Policy evaluation
        V_opt = evaluate_pi_infinite(pi_opt, P, r, gamma, max_iterations, tolerance)
        
        # Policy improvement
        for s in range(S):
            Q_values = np.zeros(A)
            for a in range(A):
                Q_values[a] = r[s, a] + gamma * np.dot(P[s, :, a], V_opt)
            pi_opt[s] = np.argmax(Q_values)
        
        # Check for convergence
        if np.array_equal(pi_opt, pi_old):
            break
    
    return pi_opt, V_opt
