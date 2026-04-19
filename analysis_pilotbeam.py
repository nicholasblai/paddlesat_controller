import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ============================================================
# Core model (same parameters as Nicholas's simulation.py)
# ============================================================

dt = 300.0
days = 30
N = int(days * 86400 / dt)
lambda_max_deg = 0.1
lambda0_deg = 0.03
v0_deg_day = 0.01
phi0 = 0.0

k_lambda = 1e-7
c_v = 3e-5
b0 = 2.5e-7
b1 = 1.2e-7
b2 = -0.7e-7
d1 = 1.0e-9
d2 = -1.2e-9
omega = 2 * np.pi / (24 * 3600)
P_max = 1.0

q_kf_lambda = 1e-18
q_kf_v      = 1e-14

def deg2rad(x): return np.deg2rad(x)
def rad2deg(x): return np.rad2deg(x)
def degday_to_rads(x): return np.deg2rad(x) / 86400.0
def rads_to_degday(x): return np.rad2deg(x) * 86400.0
def wrap_pi(a): return (a + np.pi) % (2 * np.pi) - np.pi

def b_of_phi(phi): return b0 + b1*np.cos(phi) + b2*np.sin(phi)
def d_of_phi(phi): return d1*np.cos(phi) + d2*np.sin(phi)
def alpha_to_u(a): return np.sin(2*a)
def u_to_alpha(u): return 0.5*np.arcsin(np.clip(u, -1, 1))

def transmission_efficiency(lam_rad):
    lam_max = deg2rad(lambda_max_deg)
    x = lam_rad / lam_max
    return max(0.0, 1.0 - x**2) if abs(x) <= 1.0 else 0.0

def generated_power(a): return P_max * max(0.0, np.cos(a))

def delivered_power(lam_rad, a):
    return generated_power(a) * transmission_efficiency(lam_rad)

def clipped_zenith_u(phi, clip_deg=8.0):
    return alpha_to_u(deg2rad(clip_deg) * np.sin(phi))

def step_dynamics_matrix(x_k, phi_k, u_k):
    A_d = np.array([[1.0, dt], [-k_lambda*dt, 1.0 - c_v*dt]])
    B_d = np.array([[0.0], [dt * b_of_phi(phi_k)]])
    w_k = np.array([[0.0], [dt * d_of_phi(phi_k)]])
    x_next = A_d @ x_k + B_d * u_k + w_k
    phi_next = wrap_pi(phi_k + omega * dt)
    return x_next, phi_next

# ============================================================
# Controllers
# ============================================================

def static_controller(k, x_k, phi_k, alpha_deg=0.0):
    return alpha_to_u(deg2rad(alpha_deg))

def bangbang_controller(k, x_k, phi_k, threshold_frac=0.75, clip_deg=8.0):
    lam_k = float(x_k[0, 0])
    v_k   = float(x_k[1, 0])
    lam_max = deg2rad(lambda_max_deg)
    threshold = threshold_frac * lam_max
    drifting_out = (lam_k * v_k > 0)
    if abs(lam_k) < threshold and not drifting_out:
        return 0.0
    u_corr = clipped_zenith_u(phi_k, clip_deg=clip_deg)
    return -abs(u_corr) if v_k >= 0 else abs(u_corr)

def mpc_cost(u_seq, x0, phi0, q=1.0, r=0.1, p=0.01, horizon=12):
    cost = 0.0
    x_k, phi_k = x0.copy(), phi0
    lam_max = deg2rad(lambda_max_deg)
    for i in range(horizon):
        x_next, phi_next = step_dynamics_matrix(x_k, phi_k, u_seq[i])
        li = float(x_next[0, 0])
        vi = float(x_next[1, 0])
        cost += q * (li / lam_max)**2 + r * vi**2 + p * u_seq[i]**2
        if abs(li) > lam_max:
            cost += 1000.0 * (abs(li) - lam_max)**2
        x_k, phi_k = x_next, phi_next
    return cost

def mpc_controller(k, x_k, phi_k, horizon=12, q=1.0, r=0.1, p=0.01):
    result = minimize(
        lambda u: mpc_cost(u, x_k, phi_k, q=q, r=r, p=p, horizon=horizon),
        np.zeros(horizon), method='SLSQP',
        bounds=[(-1, 1)] * horizon, options={'ftol': 1e-6, 'maxiter': 100})
    return float(result.x[0]) if result.success else 0.0

# ============================================================
# Kalman Filter
# ============================================================

class PilotBeamEstimator:
    def __init__(self, x0, sigma_pos, sigma_vel):
        self.x_hat = x0.copy()
        self.P = np.diag([sigma_pos**2, sigma_vel**2])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[sigma_pos**2]])
        self.Q = np.diag([q_kf_lambda, q_kf_v])

    def predict(self, u_k, phi_k):
        A_d = np.array([[1.0, dt], [-k_lambda*dt, 1.0 - c_v*dt]])
        B_d = np.array([[0.0], [dt * b_of_phi(phi_k)]])
        w_k = np.array([[0.0], [dt * d_of_phi(phi_k)]])
        self.x_hat = A_d @ self.x_hat + B_d * u_k + w_k
        self.P = A_d @ self.P @ A_d.T + self.Q

    def update(self, z_k):
        y = z_k - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ y
        IKH = np.eye(2) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def get_estimate(self):
        return self.x_hat.copy()

# ============================================================
# Parametric simulation runner
# ============================================================

def run_simulation(controller_name, use_pilot_beam, noise_deg=0.005,
                   mpc_q=0.10, mpc_r=0.01, mpc_p=0.0, seed=42):
    """
    Runs a full 30-day simulation and returns a dict of metrics.

    controller_name: 'static', 'bangbang', or 'mpc'
    use_pilot_beam:  True = controller sees estimated state, False = true state
    noise_deg:       pilot beam measurement noise std dev (degrees)
    """
    rng = np.random.default_rng(seed=seed)
    sigma_pos = deg2rad(noise_deg)
    sigma_vel = sigma_pos / 86400.0

    # Initialize true state
    x_hist = np.zeros((2, N + 1))
    x_hist[0, 0] = deg2rad(lambda0_deg)
    x_hist[1, 0] = degday_to_rads(v0_deg_day)
    phi_hist = np.zeros(N + 1)
    phi_hist[0] = phi0

    gen_pow = np.zeros(N)
    tx_eff  = np.zeros(N)
    del_pow = np.zeros(N)
    est_err = np.zeros((2, N))

    # Initialize Kalman filter with slightly perturbed initial estimate
    x0_est = np.array([
        [x_hist[0, 0] + rng.normal(0, sigma_pos)],
        [x_hist[1, 0] + rng.normal(0, sigma_vel)]
    ])
    kf = PilotBeamEstimator(x0_est, sigma_pos, sigma_vel)

    # Progress tracking for slow MPC runs
    show_progress = (controller_name == 'mpc')
    progress_interval = N // 10  # print every 10%
    sim_start_time = time.time()

    for k in range(N):
        # Progress indicator (MPC only)
        if show_progress and k > 0 and k % progress_interval == 0:
            pct = k * 100 // N
            day = k * dt / 86400.0
            elapsed = time.time() - sim_start_time
            eta = elapsed / k * (N - k)
            print(f"\r    [{pct:3d}%] day {day:.0f}/{days}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
                  end="", flush=True)

        x_k   = x_hist[:, [k]]
        phi_k = phi_hist[k]

        # Pilot beam measurement (noisy position reading)
        z_k = float(x_k[0, 0]) + rng.normal(0, sigma_pos)
        kf.update(np.array([[z_k]]))
        x_est = kf.get_estimate()
        est_err[:, k] = (x_k - x_est).flatten()

        # Select state the controller sees
        x_ctrl = x_est if use_pilot_beam else x_k

        # Controller dispatch
        if controller_name == 'static':
            u_k = static_controller(k, x_ctrl, phi_k, alpha_deg=0.0)
        elif controller_name == 'bangbang':
            u_k = bangbang_controller(k, x_ctrl, phi_k, clip_deg=8.0, threshold_frac=0.85)
        elif controller_name == 'mpc':
            u_k = mpc_controller(k, x_ctrl, phi_k, horizon=12, q=mpc_q, r=mpc_r, p=mpc_p)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        alpha_k = u_to_alpha(u_k)
        lam_k = float(x_k[0, 0])
        gen_pow[k] = generated_power(alpha_k)
        tx_eff[k]  = transmission_efficiency(lam_k)
        del_pow[k] = delivered_power(lam_k, alpha_k)

        # True dynamics step
        x_next, phi_next = step_dynamics_matrix(x_k, phi_k, u_k)
        x_hist[:, [k + 1]] = x_next
        phi_hist[k + 1] = phi_next

        # KF predict for next timestep
        kf.predict(u_k, phi_k)

    # Clear progress line
    if show_progress:
        print("\r" + " " * 50 + "\r", end="", flush=True)

    lam_deg = rad2deg(x_hist[0, :])

    return {
        'controller':      controller_name,
        'pilot_beam':      use_pilot_beam,
        'noise_deg':       noise_deg,
        'max_lon_err_deg': np.max(np.abs(lam_deg)),
        'avg_gen_power':   np.mean(gen_pow),
        'avg_tx_eff':      np.mean(tx_eff),
        'avg_del_power':   np.mean(del_pow),
        'frac_outside':    np.mean(np.abs(lam_deg) > lambda_max_deg),
        'rms_est_err_deg': np.sqrt(np.mean(rad2deg(est_err[0, :])**2)),
        'rms_est_err_v':   np.sqrt(np.mean(rads_to_degday(est_err[1, :])**2)),
        'lambda_deg_hist': lam_deg,
    }


# ============================================================
# PART 1: Comparison Table — 3 controllers × 2 modes
# ============================================================

print("=" * 72)
print("PART 1: Controller Comparison Table")
print("     (static and bangbang are fast, MPC takes ~4 min each)")
print("=" * 72)

configs = [
    ('static',   False),
    ('static',   True),
    ('bangbang', False),
    ('bangbang', True),
    ('mpc',      False),
    ('mpc',      True),
]

results_table = []
for ctrl, pilot in configs:
    src_label = "pilot beam" if pilot else "perfect"
    print(f"  Running: {ctrl:>8s} | {src_label:>14s} ... ", end="", flush=True)
    t0 = time.time()
    res = run_simulation(ctrl, pilot, noise_deg=0.005)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    results_table.append(res)

# Print comparison table
print("\n" + "=" * 105)
print(f"{'Controller':<12s} {'State Source':<16s} {'Max |λ| [deg]':>14s} "
      f"{'Avg Gen Pwr':>12s} {'Avg Tx Eff':>12s} {'Avg Del Pwr':>12s} "
      f"{'Frac OOB':>10s} {'RMS Est Err':>14s}")
print("-" * 105)
for r in results_table:
    src = "Pilot Beam" if r['pilot_beam'] else "Perfect"
    print(f"{r['controller']:<12s} {src:<16s} "
          f"{r['max_lon_err_deg']:>14.4f} {r['avg_gen_power']:>12.4f} "
          f"{r['avg_tx_eff']:>12.4f} {r['avg_del_power']:>12.4f} "
          f"{r['frac_outside']:>10.4f} {r['rms_est_err_deg']:>12.6f}°")
print("=" * 105)


# ============================================================
# PART 2: Noise Sensitivity Sweep (MPC controller only)
#         ~4 min per noise level
# ============================================================

print("\n" + "=" * 72)
print("PART 2: Noise Sensitivity Sweep (MPC Controller)")
print("=" * 72)

# Noise levels to sweep — add more points if you have time
noise_levels_deg = [0.001, 0.005, 0.01, 0.03, 0.05]

sweep_results = []

# Perfect-knowledge baseline
print("  Running: MPC perfect-state baseline ... ", end="", flush=True)
t0 = time.time()
baseline = run_simulation('mpc', False)
print(f"done ({time.time()-t0:.1f}s)")

for noise in noise_levels_deg:
    print(f"  Running: noise = {noise:.3f} deg ... ", end="", flush=True)
    t0 = time.time()
    res = run_simulation('mpc', True, noise_deg=noise)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    sweep_results.append(res)

# Print sweep table
print(f"\n{'Noise σ [deg]':>14s} {'Max |λ| [deg]':>14s} {'Avg Del Pwr':>12s} "
      f"{'Frac OOB':>10s} {'RMS Est Err':>14s}")
print("-" * 70)
print(f"{'Perfect':>14s} {baseline['max_lon_err_deg']:>14.4f} "
      f"{baseline['avg_del_power']:>12.4f} "
      f"{baseline['frac_outside']:>10.4f} {'N/A':>14s}")
for r in sweep_results:
    print(f"{r['noise_deg']:>14.3f} {r['max_lon_err_deg']:>14.4f} "
          f"{r['avg_del_power']:>12.4f} "
          f"{r['frac_outside']:>10.4f} {r['rms_est_err_deg']:>12.6f}°")


# ============================================================
# PLOT 1: Noise Sensitivity (2×2 grid)
# ============================================================

noise_arr       = [r['noise_deg'] for r in sweep_results]
max_lon_arr     = [r['max_lon_err_deg'] for r in sweep_results]
avg_del_pow_arr = [r['avg_del_power'] for r in sweep_results]
frac_oob_arr    = [r['frac_outside'] for r in sweep_results]
rms_est_arr     = [r['rms_est_err_deg'] for r in sweep_results]

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Pilot Beam Noise Sensitivity — MPC Controller", fontsize=14)

axs[0, 0].plot(noise_arr, max_lon_arr, 'o-', color="tab:blue", linewidth=2)
axs[0, 0].axhline(baseline['max_lon_err_deg'], color="gray", linestyle="--",
                   label="Perfect state")
axs[0, 0].axhline(lambda_max_deg, color="red", linestyle="--", alpha=0.7,
                   label="Bounding box (0.1°)")
axs[0, 0].set_xlabel("Measurement noise σ [deg]")
axs[0, 0].set_ylabel("Max |longitude error| [deg]")
axs[0, 0].set_title("Station-Keeping Accuracy")
axs[0, 0].legend(fontsize=9)
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(noise_arr, avg_del_pow_arr, 's-', color="tab:green", linewidth=2)
axs[0, 1].axhline(baseline['avg_del_power'], color="gray", linestyle="--",
                   label="Perfect state")
axs[0, 1].set_xlabel("Measurement noise σ [deg]")
axs[0, 1].set_ylabel("Average delivered power")
axs[0, 1].set_title("Power Delivery Performance")
axs[0, 1].legend(fontsize=9)
axs[0, 1].grid(True, alpha=0.3)

axs[1, 0].plot(noise_arr, [f * 100 for f in frac_oob_arr], 'D-',
               color="tab:red", linewidth=2)
axs[1, 0].axhline(baseline['frac_outside'] * 100, color="gray", linestyle="--",
                   label="Perfect state")
axs[1, 0].set_xlabel("Measurement noise σ [deg]")
axs[1, 0].set_ylabel("Time outside bounds [%]")
axs[1, 0].set_title("Bounding Box Violations")
axs[1, 0].legend(fontsize=9)
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].plot(noise_arr, [e * 1000 for e in rms_est_arr], '^-',
               color="tab:purple", linewidth=2)
axs[1, 1].set_xlabel("Measurement noise σ [deg]")
axs[1, 1].set_ylabel("RMS position est. error [mdeg]")
axs[1, 1].set_title("Estimation Quality")
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("noise_sweep.png", dpi=150)
print("\nSaved: noise_sweep.png")
plt.show()


# ============================================================
# PLOT 2: Controller comparison bar chart
# ============================================================

fig2, axs2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle("Controller Performance: Perfect State vs. Pilot Beam",
              fontsize=14)

controllers = ['static', 'bangbang', 'mpc']
labels      = ['Static', 'Bang-Bang', 'MPC']
x = np.arange(len(controllers))
width = 0.35

perfect_max = [r['max_lon_err_deg'] for r in results_table if not r['pilot_beam']]
pilot_max   = [r['max_lon_err_deg'] for r in results_table if r['pilot_beam']]
perfect_pow = [r['avg_del_power']   for r in results_table if not r['pilot_beam']]
pilot_pow   = [r['avg_del_power']   for r in results_table if r['pilot_beam']]
perfect_oob = [r['frac_outside']*100 for r in results_table if not r['pilot_beam']]
pilot_oob   = [r['frac_outside']*100 for r in results_table if r['pilot_beam']]

axs2[0].bar(x - width/2, perfect_max, width, label='Perfect State',
            color='tab:blue', alpha=0.8)
axs2[0].bar(x + width/2, pilot_max, width, label='Pilot Beam',
            color='tab:orange', alpha=0.8)
axs2[0].axhline(lambda_max_deg, color='red', linestyle='--', alpha=0.7,
                label='Bound (0.1°)')
axs2[0].set_ylabel('Max |longitude error| [deg]')
axs2[0].set_xticks(x)
axs2[0].set_xticklabels(labels)
axs2[0].legend(fontsize=8)
axs2[0].set_title('Station-Keeping')

axs2[1].bar(x - width/2, perfect_pow, width, label='Perfect State',
            color='tab:blue', alpha=0.8)
axs2[1].bar(x + width/2, pilot_pow, width, label='Pilot Beam',
            color='tab:orange', alpha=0.8)
axs2[1].set_ylabel('Average delivered power')
axs2[1].set_xticks(x)
axs2[1].set_xticklabels(labels)
axs2[1].legend(fontsize=8)
axs2[1].set_title('Power Delivery')

axs2[2].bar(x - width/2, perfect_oob, width, label='Perfect State',
            color='tab:blue', alpha=0.8)
axs2[2].bar(x + width/2, pilot_oob, width, label='Pilot Beam',
            color='tab:orange', alpha=0.8)
axs2[2].set_ylabel('Time outside bounds [%]')
axs2[2].set_xticks(x)
axs2[2].set_xticklabels(labels)
axs2[2].legend(fontsize=8)
axs2[2].set_title('Bounding Box Violations')

plt.tight_layout()
plt.savefig("controller_comparison.png", dpi=150)
print("Saved: controller_comparison.png")
plt.show()


# ============================================================
# PLOT 3: Longitude error traces — all 6 configs overlaid
# ============================================================

fig3, axs3 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig3.suptitle("Longitude Error: Perfect State vs. Pilot Beam", fontsize=14)
time_days = np.arange(N + 1) * dt / 86400.0

for idx, ctrl in enumerate(controllers):
    perf = [r for r in results_table
            if r['controller'] == ctrl and not r['pilot_beam']][0]
    pilo = [r for r in results_table
            if r['controller'] == ctrl and r['pilot_beam']][0]

    axs3[idx].plot(time_days, perf['lambda_deg_hist'],
                   label="Perfect state", alpha=0.8)
    axs3[idx].plot(time_days, pilo['lambda_deg_hist'],
                   label="Pilot beam", alpha=0.8)
    axs3[idx].axhline( lambda_max_deg, color='red', linestyle='--', alpha=0.5)
    axs3[idx].axhline(-lambda_max_deg, color='red', linestyle='--', alpha=0.5)
    axs3[idx].set_ylabel("λ error [deg]")
    axs3[idx].set_title(f"{labels[idx]} Controller")
    axs3[idx].legend(fontsize=9)
    axs3[idx].grid(True, alpha=0.2)

axs3[2].set_xlabel("Time [days]")
plt.tight_layout()
plt.savefig("controller_traces.png", dpi=150)
print("Saved: controller_traces.png")
plt.show()

print("\nDone! All plots saved to current directory.")