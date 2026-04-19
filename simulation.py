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
# SECTION 1b: Pilot Beam Parameters
# ============================================================

# The pilot beam is an uplink signal transmitted from the ground rectenna.
# The PaddleSat receives this signal and uses the received signal strength
# (RSS) to estimate its longitude error relative to the ground station.
# This replaces the assumption of perfect state knowledge.

# Pilot beam transmit power (normalized, same scale as P_max)
P_pilot = 1.0

# Pilot beam frequency (Hz) - using 5.8 GHz ISM band for the uplink
f_pilot = 5.8e9

# Pilot beam antenna beamwidth - matched to the bounding box
# The ground antenna is designed so its half-power beamwidth covers ±0.1 deg
# We model the received pilot power with the same parabolic pattern as
# the downlink transmission efficiency, since both antennas are co-located
pilot_beamwidth_deg = 0.1  # half-power at ±lambda_max

# Measurement noise standard deviation (in radians of longitude error)
# This models realistic sensor noise on the RSS-based position estimate
sigma_pilot_rad = np.deg2rad(0.005)  # 0.005 deg noise ~ good SNR at GEO

# Drift rate is not directly measured by the pilot beam, so we estimate
# it from consecutive position measurements. This has higher noise.
sigma_v_pilot = np.deg2rad(0.005) / 86400.0  # noise on drift rate estimate

# Kalman filter process noise covariance (tuning parameter)
# Represents unmodeled dynamics and disturbance uncertainty
q_kf_lambda = 1e-18   # process noise on longitude error
q_kf_v      = 1e-14   # process noise on drift rate

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
# SECTION 3b: Pilot Beam Model Functions
# ============================================================

def pilot_beam_received_power(lambda_rad):
    """
    Models the received pilot beam power at the PaddleSat as a function
    of longitude error. Uses the same parabolic beam pattern as the
    downlink transmission efficiency, since both ground antennas are
    co-located at the rectenna site.
    
    Returns normalized received power in [0, 1].
    """
    lam_max = deg2rad(pilot_beamwidth_deg)
    x = lambda_rad / lam_max
    if abs(x) <= 1.0:
        return max(0.0, P_pilot * (1.0 - x**2))
    else:
        return 0.0

def pilot_beam_measurement(lambda_true_rad, rng=None):
    """
    Simulates a noisy longitude error measurement derived from the
    pilot beam RSS. The satellite measures the received pilot power
    and inverts the beam pattern model to estimate its longitude error.
    
    In practice, the satellite would:
    1. Measure the received pilot beam power
    2. Compare it to the known transmitted power
    3. Invert the beam pattern equation to get longitude error
    
    We model this as: z = lambda_true + noise
    """
    if rng is None:
        rng = np.random.default_rng()
    
    noise = rng.normal(0.0, sigma_pilot_rad)
    return lambda_true_rad + noise

def pilot_beam_snr(lambda_rad):
    """
    Estimates the pilot beam SNR as a function of position.
    Higher SNR near center of beam -> lower measurement noise.
    Returns SNR in linear scale (not dB).
    """
    p_rx = pilot_beam_received_power(lambda_rad)
    # Noise floor normalized to give ~20 dB SNR at beam center
    noise_floor = P_pilot / 100.0
    if p_rx > 0:
        return p_rx / noise_floor
    else:
        return 0.0

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
# SECTION 5b: Pilot Beam State Estimator (Kalman Filter)
# ============================================================

class PilotBeamEstimator:
    """
    A Kalman Filter that fuses the pilot beam position measurement
    with the known dynamics model to estimate the full state [lambda, v].
    
    The pilot beam only gives us a (noisy) longitude error measurement.
    The Kalman filter uses the dynamics model to also estimate drift rate,
    which is not directly observable from a single pilot beam reading.
    """
    
    def __init__(self, x0, P0=None):
        """
        x0: initial state estimate [lambda, v] as (2,1) array
        P0: initial covariance (2x2), defaults to identity scaled
        """
        self.x_hat = x0.copy()
        
        if P0 is None:
            self.P = np.array([
                [sigma_pilot_rad**2, 0.0],
                [0.0, sigma_v_pilot**2]
            ])
        else:
            self.P = P0.copy()
        
        # Measurement matrix: we only measure longitude error (first state)
        self.H = np.array([[1.0, 0.0]])
        
        # Measurement noise covariance
        self.R = np.array([[sigma_pilot_rad**2]])
        
        # Process noise covariance
        self.Q = np.array([
            [q_kf_lambda, 0.0],
            [0.0, q_kf_v]
        ])
    
    def predict(self, u_k, phi_k):
        """
        Prediction step: propagate state estimate using dynamics model.
        """
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
        
        # State prediction
        self.x_hat = A_d @ self.x_hat + B_d * u_k + w_k
        
        # Covariance prediction
        self.P = A_d @ self.P @ A_d.T + self.Q
    
    def update(self, z_k):
        """
        Update step: correct the prediction using pilot beam measurement.
        z_k: scalar measurement of longitude error
        """
        # Innovation (measurement residual)
        y = z_k - self.H @ self.x_hat
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x_hat = self.x_hat + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(2)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
    
    def get_estimate(self):
        """Returns current state estimate as (2,1) array."""
        return self.x_hat.copy()

# ============================================================
# SECTION 5c: Pilot Beam Aware Controllers
# ============================================================

def pilotbeam_mpc_controller(k, x_est, phi_k, horizon=12, q=1.0, r=0.1, p=0.01):
    """
    MPC controller that operates on the estimated state from the pilot
    beam Kalman filter, rather than the true state.
    
    This is the same MPC algorithm, but it demonstrates the realistic
    scenario where the satellite doesn't have perfect state knowledge -
    it must rely on the pilot beam for position feedback.
    """
    return mpc_controller(k, x_est, phi_k, horizon=horizon, q=q, r=r, p=p)

def pilotbeam_bangbang_controller(k, x_est, phi_k, threshold_frac=0.75, clip_deg=8.0):
    """
    Bang-bang controller that operates on pilot beam estimated state.
    """
    return bangbang_controller(k, x_est, phi_k, threshold_frac=threshold_frac, clip_deg=clip_deg)

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

# Pilot beam histories
x_est_hist           = np.zeros((2, N+1))   # Kalman filter state estimates
pilot_meas_hist      = np.zeros(N)          # raw pilot beam measurements
pilot_rx_power_hist  = np.zeros(N)          # received pilot beam power
pilot_snr_hist       = np.zeros(N)          # pilot beam SNR
estimation_error_hist = np.zeros((2, N))    # estimation error (true - estimated)

x_hist[0, 0] = deg2rad(lambda0_deg)
x_hist[1, 0] = degday_to_rads(v0_deg_day)
phi_hist[0]  = phi0

# Initialize the Kalman filter with a slightly perturbed initial estimate
# (the satellite doesn't know its exact state at startup)
rng = np.random.default_rng(seed=42)
x0_est = np.array([
    [deg2rad(lambda0_deg) + rng.normal(0, sigma_pilot_rad)],
    [degday_to_rads(v0_deg_day) + rng.normal(0, sigma_v_pilot)]
])
kf = PilotBeamEstimator(x0_est)
x_est_hist[:, [0]] = x0_est

# ============================================================
# SECTION 7: Main simulation loop
# ============================================================

# Set USE_PILOT_BEAM to True to use the pilot beam estimator for control,
# or False to use perfect state knowledge (original behavior).
USE_PILOT_BEAM = True

for k in range(N):
    x_k   = x_hist[:, [k]]        # true state
    phi_k = phi_hist[k]

    # --- Pilot Beam Measurement ---
    lambda_true = float(x_k[0, 0])
    z_k = pilot_beam_measurement(lambda_true, rng=rng)
    pilot_meas_hist[k] = z_k
    pilot_rx_power_hist[k] = pilot_beam_received_power(lambda_true)
    pilot_snr_hist[k] = pilot_beam_snr(lambda_true)

    # --- Kalman Filter Update ---
    # Update the estimator with the new pilot beam measurement
    kf.update(np.array([[z_k]]))
    x_est = kf.get_estimate()

    # Record estimation error
    estimation_error_hist[:, k] = (x_k - x_est).flatten()

    # --- Controller ---
    if USE_PILOT_BEAM:
        # Controller uses ESTIMATED state from pilot beam + Kalman filter
        # CHOOSE YOUR PILOT BEAM CONTROLLER HERE
        u_k = pilotbeam_mpc_controller(k, x_est, phi_k, horizon=12, q=0.10, r=0.01, p=0.0)
        # u_k = pilotbeam_bangbang_controller(k, x_est, phi_k, clip_deg=8.0, threshold_frac=0.85)
    else:
        # Original: controller uses TRUE state (perfect knowledge)
        # CHOOSE YOUR CONTROLLER HERE
        # u_k = static_controller(k, x_k, phi_k, alpha_deg=0.0)
        # u_k = bangbang_controller(k, x_k, phi_k, clip_deg=8.0, threshold_frac=0.85)
        u_k = mpc_controller(k, x_k, phi_k, horizon=12, q=0.10, r=0.01, p=0.0)

    alpha_k = u_to_alpha(u_k)

    # Save histories for plotting
    u_hist[k]     = u_k
    alpha_hist[k] = alpha_k

    lambda_k = x_k[0, 0]
    gen_power_hist[k] = generated_power(alpha_k)
    tx_eff_hist[k] = transmission_efficiency(lambda_k)
    delivered_power_hist[k] = delivered_power(lambda_k, alpha_k)

    # --- Dynamics Step (true system) ---
    x_next, phi_next = step_dynamics_matrix(x_k, phi_k, u_k)

    x_hist[:, [k+1]] = x_next
    phi_hist[k+1]    = phi_next

    # --- Kalman Filter Predict (for next timestep) ---
    kf.predict(u_k, phi_k)
    x_est_hist[:, [k+1]] = kf.get_estimate()

time_days        = np.arange(N+1) * dt / 86400.0
lambda_deg_hist  = rad2deg(x_hist[0, :])
v_degday_hist    = rads_to_degday(x_hist[1, :])
alpha_deg_hist   = rad2deg(alpha_hist)

# Pilot beam estimated state histories
lambda_est_deg_hist = rad2deg(x_est_hist[0, :])
v_est_degday_hist   = rads_to_degday(x_est_hist[1, :])


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

# Pilot beam estimation metrics
est_err_lambda_deg = rad2deg(estimation_error_hist[0, :])
est_err_v_degday   = rads_to_degday(estimation_error_hist[1, :])
avg_pilot_snr_db   = 10 * np.log10(np.mean(pilot_snr_hist[pilot_snr_hist > 0]) + 1e-30)

print(f"\n=== Pilot Beam Metrics ===")
print(f"Controller uses pilot beam: {USE_PILOT_BEAM}")
print(f"RMS position est. error   : {np.sqrt(np.mean(est_err_lambda_deg**2)):.6f} deg")
print(f"RMS drift rate est. error : {np.sqrt(np.mean(est_err_v_degday**2)):.6f} deg/day")
print(f"Max |position est. error| : {np.max(np.abs(est_err_lambda_deg)):.6f} deg")
print(f"Avg pilot beam SNR        : {avg_pilot_snr_db:.1f} dB")
print(f"Avg pilot rx power        : {np.mean(pilot_rx_power_hist):.4f}")


# ============================================================
# SECTION 10: Plot results
# ============================================================

fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

axs[0].plot(time_days, lambda_deg_hist, label="True longitude error")
axs[0].plot(time_days, lambda_est_deg_hist, label="Estimated (pilot beam)", alpha=0.7, linestyle="--")
axs[0].axhline( lambda_max_deg, linestyle="--", color="red", label="Bounds")
axs[0].axhline(-lambda_max_deg, linestyle="--", color="red")
axs[0].set_ylabel("Longitude error [deg]")
title_suffix = " (Pilot Beam + KF)" if USE_PILOT_BEAM else " (Perfect State)"
axs[0].set_title(f"Reduced-Order PaddleSat Simulation{title_suffix}")
axs[0].legend()

axs[1].plot(time_days, v_degday_hist, label="True drift rate")
axs[1].plot(time_days, v_est_degday_hist, label="Estimated (pilot beam)", alpha=0.7, linestyle="--")
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
plt.savefig("simulation_main.png", dpi=150)
plt.show()

# ============================================================
# SECTION 11: Pilot Beam Diagnostic Plots
# ============================================================

fig2, axs2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Estimation error
axs2[0].plot(time_days[:-1], est_err_lambda_deg * 1000, label="Position est. error", color="tab:red")
axs2[0].set_ylabel("Position error [mdeg]")
axs2[0].set_title("Pilot Beam Estimation Diagnostics")
axs2[0].legend()
axs2[0].axhline(0, color="gray", linestyle=":", alpha=0.5)

axs2[1].plot(time_days[:-1], est_err_v_degday, label="Drift rate est. error", color="tab:orange")
axs2[1].set_ylabel("Drift rate error [deg/day]")
axs2[1].legend()
axs2[1].axhline(0, color="gray", linestyle=":", alpha=0.5)

# Pilot beam received power
axs2[2].plot(time_days[:-1], pilot_rx_power_hist, label="Pilot Rx power", color="tab:green")
axs2[2].set_ylabel("Received power")
axs2[2].legend()

# Pilot beam SNR
axs2[3].plot(time_days[:-1], 10 * np.log10(pilot_snr_hist + 1e-30), label="Pilot SNR", color="tab:blue")
axs2[3].set_ylabel("SNR [dB]")
axs2[3].set_xlabel("Time [days]")
axs2[3].legend()

plt.tight_layout()
plt.savefig("pilot_beam_diagnostics.png", dpi=150)
plt.show()
