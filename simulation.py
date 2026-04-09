import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# SECTION 1: Simulation parameters
# ============================================================

# Time Parameters
dt = 300.0
days = 30
N = int(days * 86400 / dt)

# Maximum Orbital Drift in Degrees
lambda_max_deg = 0.1

# Initial conditions
lambda0_deg = 0.03
v0_deg_day = 0.01
phi0 = 0.0

# Reduced-order model parameters
k_lambda = 1e-7                # weak restoring term
c_v = 3e-5                     # damping on drift rate

# Phase-dependent control effectiveness:
# b(phi) = b0 + b1 * cos(phi) + b2 * sin(phi)
b0 = 2.5e-7
b1 = 1.2e-7
b2 = -0.7e-7

# Phase-dependent disturbance:
# d(phi) = d1 * cos(phi) + d2 * sin(phi)
d1 = 1.0e-9
d2 = -1.2e-9

# One periodic cycle per day
omega = 2 * np.pi / (24 * 3600)

# Maximum normalized power
P_max = 1.0

# Control Input Constraints
U_MAX = 1.0
U_MIN = -1.0

# ============================================================
# SECTION 2: Unit-conversion helper functions
# ============================================================

def deg2rad(x_deg):
    return np.deg2rad(x_deg)

def rad2deg(x_rad):
    return np.rad2deg(x_rad)

def degday_to_rads(x_deg_day):
    return np.deg2rad(x_deg_day) / 86400.0

def rads_to_degday(x_rad_s):
    return np.rad2deg(x_rad_s) * 86400.0

def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ============================================================
# SECTION 3: Model functions
# ============================================================

# Models how the phase affects the control input
def b_of_phi(phi):
    return b0 + b1 * np.cos(phi) + b2 * np.sin(phi)

# Models how the phase affects the disturbance
def d_of_phi(phi):
    return d1 * np.cos(phi) + d2 * np.sin(phi)

# Models the control input as u_k = sin(2 * alpha), where alpha is the angle of solar incidence
def alpha_to_u(alpha_rad):
    return np.sin(2 * alpha_rad)

# Recovers the angle from the linearized control variable
def u_to_alpha(u):
    return 0.5 * np.arcsin(np.clip(u, -1.0, 1.0))

# Calculates the transmission efficiency, t, which can be expressed as a parabola 
def transmission_efficiency(lambda_rad):
    lam_max = deg2rad(lambda_max_deg)
    x = lambda_rad / lam_max
    if abs(x) <= 1.0:
        return max(0.0, 1.0 - x**2)
    else:
        return 0.0

# Calculates the generated solar power (efficiency * peak power generation), P * s, which can be modeled with a cos
def generated_power(alpha_rad):
    return P_max * max(0.0, np.cos(alpha_rad))

# Calculates total power = P * s * t
def delivered_power(lambda_rad, alpha_rad):
    return generated_power(alpha_rad) * transmission_efficiency(lambda_rad)

# Helper function for the BangBang Controller, mimics the clipped zenith as a function of phi 
def clipped_zenith_u(phi_k, clip_deg=8.0):
    clip_rad = deg2rad(clip_deg)
    alpha_raw = clip_rad * np.sin(phi_k)
    return alpha_to_u(alpha_raw)

# ============================================================
# SECTION 4: Matrix-form state space update model
# ============================================================

def step_dynamics_matrix(x_k, phi_k, u_k):
    A_d = np.array([
        [1.0, dt],
        [-k_lambda * dt, 1.0 - c_v * dt]
    ])

    B_d = np.array([
        [0.0],
        [dt * b_of_phi(phi_k)]
    ])

    w_k = np.array([
        [0.0],
        [dt * d_of_phi(phi_k)]
    ])

    x_next = A_d @ x_k + B_d * u_k + w_k

    phi_next = wrap_pi(phi_k + omega * dt)

    return x_next, phi_next

# ============================================================
# SECTION 5: Controllers
# ============================================================

# Static controller that maintains a constant angle of solar incidence
def static_controller(k, x_k, phi_k, alpha_deg=0.0):
    alpha_rad = deg2rad(alpha_deg)
    return alpha_to_u(alpha_rad)   # returns u in [-1, 1]

# Bang-Bang controller implementation
def bangbang_controller(k, x_k, phi_k, threshold_frac=0.75, clip_deg=8.0):
    lambda_k = float(x_k[0, 0])
    v_k      = float(x_k[1, 0])
    lam_max  = deg2rad(lambda_max_deg)
    threshold = threshold_frac * lam_max

    drifting_outward = (lambda_k * v_k > 0)

    # If the satellite is not at the 75% boundary and not drifting outward, then coast, which maximizes power
    if abs(lambda_k) < threshold and not drifting_outward:
        return 0.0

    # If the satellite is at the boundary and drifting outward, then apply a correction, which is a phase dependent term, meant to model how the sun is not always in the same spot
    u_correction = clipped_zenith_u(phi_k, clip_deg=clip_deg)
    if v_k >= 0:
        return -abs(u_correction)   # push west
    else:
        return abs(u_correction)   # push east

# Cost function for the Model Predictive Control
def mpc_cost(u_sequence, x0, phi0, q=1.0, r=0.1, p=0.01, horizon=12):
    cost = 0.0
    x_k = x0.copy()
    phi_k = phi0

    lam_max = deg2rad(lambda_max_deg)

    for i in range(horizon):
        u_k = u_sequence[i]

        # Step the model forward
        x_next, phi_next = step_dynamics_matrix(x_k, phi_k, u_k)

        lambda_i = float(x_next[0, 0])
        v_i      = float(x_next[1, 0])

        # State cost: penalize drift rate and being outside the box
        lambda_normalized = lambda_i / lam_max
        cost += q * lambda_normalized**2
        cost += r * v_i**2
        
        # Further penalize being outside of lambda_max
        if abs(lambda_i) > lam_max:
            cost += 1000.0 * (abs(lambda_i) - lam_max)**2

        # Control cost: penalize using control effort, which pushes you away from max power generation (alpha = 0)
        cost += p * u_k**2

        # Advance
        x_k   = x_next
        phi_k = phi_next
    
    return cost

def mpc_controller(k, x_k, phi_k, horizon=12, q=1.0, r=0.1, p=0.01):
    # Initial Guess
    u_init = np.zeros(horizon)

    bounds = [(-1.0, 1.0)] * horizon

    # Calculate the cost
    def objective(u_sequence):
        return mpc_cost(u_sequence, x_k, phi_k,
                            q=q, r=r, p=p, horizon=horizon)

    # Find the path with the minimum cost 
    result = minimize(
        objective,
        u_init,
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 100}
    )

    # Apply the first control from the minimum cost path
    if result.success:
        return float(result.x[0])
    else:
        return 0.0

# ============================================================
# SECTION 6: Allocate arrays and initialize simulation
# ============================================================

x_hist            = np.zeros((2, N+1))
phi_hist          = np.zeros(N+1)

u_hist            = np.zeros(N)
alpha_hist        = np.zeros(N)

gen_power_hist       = np.zeros(N)
tx_eff_hist          = np.zeros(N)
delivered_power_hist = np.zeros(N)

x_hist[0, 0] = deg2rad(lambda0_deg)
x_hist[1, 0] = degday_to_rads(v0_deg_day)
phi_hist[0]  = phi0

# ============================================================
# SECTION 7: Main simulation loop
# ============================================================

for k in range(N):
    x_k   = x_hist[:, [k]]
    phi_k = phi_hist[k]

    # CHOOSE YOUR CONTROLLER HERE
    # u_k = static_controller(k, x_k, phi_k, alpha_deg=0.0)
    # u_k = bangbang_controller(k, x_k, phi_k, clip_deg=8.0, threshold_frac=0.85)
    # MPC CONTROLLERS
    # Very agressive MPC controller focused on minimizing longitude error/station-keeping and less on power generation
    u_k = mpc_controller(k, x_k, phi_k, horizon=12, q=0.10, r=0.01, p=0.0)

    # Less agressive MPC Controller, focused more on power generation and less on station-keeping, allowing for more drift
    # u_k = mpc_controller(k, x_k, phi_k, horizon=12, q=0.000001, r=0.00001, p=1.0)

    alpha_k = u_to_alpha(u_k)

    # Save histories for plotting
    u_hist[k]     = u_k
    alpha_hist[k] = alpha_k

    lambda_k = x_k[0, 0]
    gen_power_hist[k] = generated_power(alpha_k)
    tx_eff_hist[k] = transmission_efficiency(lambda_k)
    delivered_power_hist[k] = delivered_power(lambda_k, alpha_k)

    x_next, phi_next = step_dynamics_matrix(x_k, phi_k, u_k)

    x_hist[:, [k+1]] = x_next
    phi_hist[k+1]    = phi_next

time_days        = np.arange(N+1) * dt / 86400.0
lambda_deg_hist  = rad2deg(x_hist[0, :])
v_degday_hist    = rads_to_degday(x_hist[1, :])
alpha_deg_hist   = rad2deg(alpha_hist)


# ============================================================
# SECTION 8: Print summary metrics
# ============================================================

max_abs_lambda_deg   = np.max(np.abs(lambda_deg_hist))
avg_gen_power        = np.mean(gen_power_hist)
avg_tx_eff           = np.mean(tx_eff_hist)
avg_delivered_power  = np.mean(delivered_power_hist)
frac_outside_bounds  = np.mean(np.abs(lambda_deg_hist) > lambda_max_deg)

print("=== Summary Metrics ===")
print(f"Max |longitude error| [deg]: {max_abs_lambda_deg:.4f}")
print(f"Average generated power   : {avg_gen_power:.4f}")
print(f"Average transmission eff. : {avg_tx_eff:.4f}")
print(f"Average delivered power   : {avg_delivered_power:.4f}")
print(f"Fraction outside bounds   : {frac_outside_bounds:.4f}")


# ============================================================
# SECTION 10: Plot results
# ============================================================

fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

axs[0].plot(time_days, lambda_deg_hist, label="Longitude error")
axs[0].axhline( lambda_max_deg, linestyle="--", color="red", label="Bounds")
axs[0].axhline(-lambda_max_deg, linestyle="--", color="red")
axs[0].set_ylabel("Longitude error [deg]")
axs[0].set_title("Reduced-Order PaddleSat Simulation")
axs[0].legend()

axs[1].plot(time_days, v_degday_hist, label="Drift rate")
axs[1].set_ylabel("Drift rate [deg/day]")
axs[1].legend()

# Raw decision variable u_k
axs[2].plot(time_days[:-1], u_hist, label="u = sin(2α)", color="purple")
axs[2].axhline( U_MAX, linestyle="--", color="gray", label="u bounds")
axs[2].axhline( U_MIN, linestyle="--", color="gray")
axs[2].set_ylabel("Control input u")
axs[2].legend()

# Recovered alpha (for physical interpretation)
axs[3].plot(time_days[:-1], alpha_deg_hist, label="Recovered α [deg]",
            color="orange", linestyle="--")
axs[3].set_ylabel("Panel angle α [deg]")
axs[3].legend()

axs[4].plot(time_days[:-1], delivered_power_hist, label="Delivered power")
axs[4].set_ylabel("Delivered power")
axs[4].set_xlabel("Time [days]")
axs[4].legend()

plt.tight_layout()
plt.show()