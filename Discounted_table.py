import numpy as np
from scipy.stats import binom

# Baseline calibration (Table 1); delta = 0.99
alpha, beta, c, gamma = 100.0, 10.0, 0.01, 0.5
q, p, delta = 0.8, 0.01, 0.99
ETAS = [0.50, 0.20, 0.10, 0.05, -0.05, -0.10, -0.20]

X_MAX = 120                     # truncation; all reported thresholds <= 81
xgrid = np.arange(X_MAX + 1)
S = beta * xgrid - alpha        # stopping reward S(x)

# One-step atoms: A_x ~ Binomial(x,p), D ~ Bernoulli(q), next = x - A_x + D
atoms = []
for x in xgrid:
    a = np.arange(x + 1)
    pa = binom.pmf(a, x, p)
    atoms.append((np.concatenate([a, a]),
                  np.concatenate([pa * (1 - q), pa * q]),
                  np.concatenate([x - a, x - a + 1])))

def extend(V):                  # imposed stopping boundary at X_MAX + 1
    return np.concatenate([V, [beta * (X_MAX + 1) - alpha]])

# Risk-neutral benchmark (also Method A's terminal condition)
def risk_neutral_value(tol=1e-12):
    C_eff = c + delta * gamma * p
    V = S.copy()
    for _ in range(50000):
        V_ext = extend(V)
        cont = np.array([-C_eff * x + delta * np.dot(atoms[x][1], V_ext[atoms[x][2]])
                         for x in xgrid])
        V_new = np.maximum(S, cont)
        if np.max(np.abs(V_new - V)) < tol: break
        V = V_new
    return V, int(np.min(xgrid[V <= S + 1e-9]))

# Method A: backward recursion on the grid lam_k = eta*delta^k,
# terminal W(x, lam_K) = exp(lam_K * V_rn(x)); opt = max (eta>0) / min (eta<0)
def method_A(eta, Vrn, lam_tol=1e-5):
    K = int(np.ceil(np.log(lam_tol / abs(eta)) / np.log(delta)))
    W = np.exp(eta * delta**K * extend(Vrn))
    x_star = None
    for k in range(K, -1, -1):
        lam = eta * delta**k
        stop = np.exp(lam * S)
        cont = np.array([np.dot(atoms[x][1],
                np.exp(-lam * (c * x + delta * gamma * atoms[x][0])) * W[atoms[x][2]])
                         for x in xgrid])
        W_new = np.maximum(stop, cont) if eta > 0 else np.minimum(stop, cont)
        if k == 0:
            picked = (stop >= cont) if eta > 0 else (stop <= cont)
            x_star = int(np.min(xgrid[picked]))
        W = extend(W_new)
    return np.log(W[0]) / eta, x_star

# Method B: stationary recursive-entropic Bellman, delta-contraction
def method_B(eta, tol=1e-11):
    V = S.copy()
    for _ in range(100000):
        V_ext = extend(V)
        cont = np.array([-c * x + np.log(np.dot(atoms[x][1],
                np.exp(eta * delta * (V_ext[atoms[x][2]] - gamma * atoms[x][0])))) / eta
                         for x in xgrid])
        V_new = np.maximum(S, cont)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    return V[0], int(np.min(xgrid[V <= S + 1e-9]))

if __name__ == "__main__":
    Vrn, theta_rn = risk_neutral_value()
    print(f"Risk-neutral benchmark: V(0) = {Vrn[0]:.4f}, theta*_rn = {theta_rn}")
    for eta in ETAS:
        vA, xA = method_A(eta, Vrn)
        vB, thB = method_B(eta)
        print(f"{eta:+.2f} & {vA:.2f} & {xA} & {vB:.2f} & {thB} \\\\")
