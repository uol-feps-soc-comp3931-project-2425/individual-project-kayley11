import numpy as np
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import numba
from numba import jit, prange, float64, int64
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PerformanceOptimization")

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self.start_times = {}

    def start_timer(self, name):
        self.start_times[name] = time.time()

    def stop_timer(self, name):
        if name not in self.start_times:
            logger.warning(f"计时器{name}未启动")
            return 0
        elapsed = time.time() - self.start_times[name]
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        self.timings[name].append(elapsed)
        self.call_counts[name] += 1
        return elapsed

    def get_average_time(self, name):
        if name not in self.timings or len(self.timings[name]) == 0:
            return 0
        return sum(self.timings[name]) / len(self.timings[name])

    def get_total_time(self, name):
        if name not in self.timings:
            return 0
        return sum(self.timings[name])

    def get_call_count(self, name):
        if name not in self.call_counts:
            return 0
        return self.call_counts[name]

    def reset(self):
        self.timings = {}
        self.call_counts = {}
        self.start_times = {}

    def get_performance_report(self):
        report = "性能报告:\n"
        if not self.timings:
            return report + "尚无性能数据"
        sorted_names = sorted(self.timings.keys(),
                             key=lambda name: sum(self.timings[name]),
                             reverse=True)
        for name in sorted_names:
            total = sum(self.timings[name])
            avg = total / len(self.timings[name])
            calls = self.call_counts[name]
            report += f"{name}:\n"
            report += f"  调用次数: {calls}\n"
            report += f"  总耗时: {total:.6f}秒\n"
            report += f"  平均耗时: {avg:.6f}秒/调用\n"
            if len(self.timings[name]) > 1:
                min_time = min(self.timings[name])
                max_time = max(self.timings[name])
                report += f"  最短耗时: {min_time:.6f}秒\n"
                report += f"  最长耗时: {max_time:.6f}秒\n"
            report += "\n"
        return report

    def plot_performance(self, output_dir="performance_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.timings:
            logger.warning("尚无性能数据，无法绘图")
            return
        plt.figure(figsize=(12, 6))
        names = list(self.timings.keys())
        total_times = [sum(self.timings[name]) for name in names]
        sorted_indices = np.argsort(total_times)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_times = [total_times[i] for i in sorted_indices]
        plt.bar(sorted_names, sorted_times)
        plt.xlabel('函数/操作')
        plt.ylabel('总耗时 (秒)')
        plt.title('各操作总耗时对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "total_time.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(12, 6))
        avg_times = [sum(self.timings[name])/len(self.timings[name]) for name in names]
        sorted_indices = np.argsort(avg_times)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_avg_times = [avg_times[i] for i in sorted_indices]
        plt.bar(sorted_names, sorted_avg_times)
        plt.xlabel('函数/操作')
        plt.ylabel('平均耗时 (秒/调用)')
        plt.title('各操作平均耗时对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "average_time.png"), dpi=300)
        plt.close()
        for name in names:
            if len(self.timings[name]) > 5:
                plt.figure(figsize=(10, 5))
                plt.hist(self.timings[name], bins=20, alpha=0.7)
                plt.xlabel('耗时 (秒)')
                plt.ylabel('频次')
                plt.title(f'{name} 耗时分布')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_distribution.png"), dpi=300)
                plt.close()

@jit(float64[:](float64, float64[:], float64, float64, float64), nopython=True, cache=True)
def lorenz_derivatives_optimized(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

@jit(float64[:](float64, float64[:], float64, float64, float64, float64), nopython=True, cache=True)
def rk4_step_optimized(t, state, dt, sigma, rho, beta):
    k1 = lorenz_derivatives_optimized(t, state, sigma, rho, beta)
    k2 = lorenz_derivatives_optimized(t + 0.5 * dt, state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_derivatives_optimized(t + 0.5 * dt, state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_derivatives_optimized(t + dt, state + dt * k3, sigma, rho, beta)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

@jit(nopython=True, cache=True)
def solve_lorenz_optimized(t_start, t_end, initial_state, dt, sigma, rho, beta):
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, 3))
    states[0] = initial_state
    for i in range(1, n_steps):
        states[i] = rk4_step_optimized(t[i-1], states[i-1], dt, sigma, rho, beta)
    return t, states

def solve_batch_parallel(param_sets, t_span, initial_state, dt, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    t_start, t_end = t_span
    def solve_single(idx, params):
        sigma, rho, beta = params
        t, states = solve_lorenz_optimized(t_start, t_end, initial_state, dt, sigma, rho, beta)
        return idx, (t, states)
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(solve_single, i, params) for i, params in enumerate(param_sets)]
        for future in futures:
            idx, result = future.result()
            results[idx] = result
    return results

def solve_batch_initial_conditions(system, solver, initial_conditions, t_span, dt):
    initial_array = np.array(initial_conditions)
    n_conditions = len(initial_conditions)
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    all_states = np.zeros((n_conditions, n_steps, 3))
    all_states[:, 0, :] = initial_array
    for i in range(1, n_steps):
        current_t = t[i-1]
        current_states = all_states[:, i-1, :]
        for j in range(n_conditions):
            all_states[j, i, :] = solver.step(current_t, current_states[j], dt)
    results = {}
    for j in range(n_conditions):
        results[j] = (t, all_states[j])
    return results

class OptimizedStructurePreservingIntegrator:
    def __init__(self, system=None, enhanced=False):
        self.system = system
        self.sigma = system.sigma if system else 10.0
        self.rho = system.rho if system else 28.0
        self.beta = system.beta if system else 8/3
        self.enhanced = enhanced
        self.theoretical_contraction_rate = -(self.sigma + 1 + self.beta)
        self.performance_monitor = PerformanceMonitor()
        self.stats = {
            'volume_corrections': 0,
            'constraint_enforcements': 0,
            'attractor_preservations': 0,
            'total_steps': 0
        }

    @staticmethod
    @jit(float64[:](float64, float64[:], float64, float64, float64, float64), nopython=True, cache=True)
    def _step_core(t, state, dt, sigma, rho, beta):
        k1 = lorenz_derivatives_optimized(t, state, sigma, rho, beta)
        k2 = lorenz_derivatives_optimized(t + dt/2, state + dt*k1/2, sigma, rho, beta)
        k3 = lorenz_derivatives_optimized(t + dt/2, state + dt*k2/2, sigma, rho, beta)
        k4 = lorenz_derivatives_optimized(t + dt, state + dt*k3, sigma, rho, beta)
        next_state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_state

    @staticmethod
    @jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)
    def _apply_volume_contraction_correction(state, next_state, dt, jacobian_trace):
        state_norm = np.sqrt(np.sum(state**2))
        next_state_norm = np.sqrt(np.sum(next_state**2))
        if state_norm > 1e-10:
            numerical_contraction = (next_state_norm - state_norm) / (dt * state_norm)
        else:
            numerical_contraction = 0.0
        correction_factor = np.exp(dt * (jacobian_trace - numerical_contraction))
        corrected_state = next_state * correction_factor
        return corrected_state

    @staticmethod
    @jit(float64[:](float64[:], float64), nopython=True, cache=True)
    def _apply_constraint_enforcement(state, max_value):
        x, y, z = state
        if z < 0:
            z = 1e-6
        max_abs = max(abs(x), abs(y), abs(z))
        if max_abs > max_value:
            scale_factor = max_value / max_abs
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
        return np.array([x, y, z])

    @staticmethod
    @jit(float64[:](float64[:], float64[:], float64, float64, float64, boolean), nopython=True, cache=True)
    def _apply_attractor_preservation(state, next_state, dt, rho, energy_change_threshold, enhanced):
        x, y, z = state
        next_x, next_y, next_z = next_state
        energy = x**2 + y**2 + (z - rho)**2
        next_energy = next_x**2 + next_y**2 + (next_z - rho)**2
        if energy > 1e-10:
            energy_change_rate = (next_energy - energy) / (dt * energy)
        else:
            energy_change_rate = 0.0
        if enhanced:
            if abs(energy_change_rate) > energy_change_threshold:
                if next_energy > 1e-10:
                    correction_factor = np.sqrt(energy / next_energy)
                    next_state = next_state * correction_factor
        else:
            if energy_change_rate > energy_change_threshold:
                if next_energy > 1e-10:
                    correction_factor = np.sqrt(energy / next_energy)
                    next_state = next_state * correction_factor
        return next_state

    def step(self, t, state, dt):
        self.stats['total_steps'] += 1
        self.performance_monitor.start_timer("step_total")
        self.performance_monitor.start_timer("step_core")
        next_state = self._step_core(t, state, dt, self.sigma, self.rho, self.beta)
        self.performance_monitor.stop_timer("step_core")
        self.performance_monitor.start_timer("structure_preserving_corrections")
        next_state = self.apply_structure_preserving_corrections(state, next_state, dt)
        self.performance_monitor.stop_timer("structure_preserving_corrections")
        self.performance_monitor.stop_timer("step_total")
        return next_state

    def apply_structure_preserving_corrections(self, state, next_state, dt):
        self.performance_monitor.start_timer("volume_contraction_correction")
        next_state = self.apply_volume_contraction_correction(state, next_state, dt)
        self.performance_monitor.stop_timer("volume_contraction_correction")
        self.performance_monitor.start_timer("constraint_enforcement")
        next_state = self.apply_constraint_enforcement(next_state)
        self.performance_monitor.stop_timer("constraint_enforcement")
        self.performance_monitor.start_timer("attractor_preservation")
        next_state = self.apply_attractor_preservation(state, next_state, dt)
        self.performance_monitor.stop_timer("attractor_preservation")
        return next_state

    def apply_volume_contraction_correction(self, state, next_state, dt):
        jacobian_trace = self.theoretical_contraction_rate
        corrected_state = self._apply_volume_contraction_correction(
            state, next_state, dt, jacobian_trace
        )
        self.stats['volume_corrections'] += 1
        return corrected_state

    def apply_constraint_enforcement(self, state):
        max_value = 1000.0
        constrained_state = self._apply_constraint_enforcement(state, max_value)
        if not np.array_equal(state, constrained_state):
            self.stats['constraint_enforcements'] += 1
        return constrained_state

    def apply_attractor_preservation(self, state, next_state, dt):
        energy_change_threshold = 0.1 if self.enhanced else 0.2
        preserved_state = self._apply_attractor_preservation(
            state, next_state, dt, self.rho, energy_change_threshold, self.enhanced
        )
        if not np.array_equal(next_state, preserved_state):
            self.stats['attractor_preservations'] += 1
        return preserved_state

    def solve(self, t_span, initial_state, dt):
        self.performance_monitor.start_timer("solve_total")
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)
        states = np.zeros((n_steps, 3))
        states[0] = initial_state
        self.stats = {
            'volume_corrections': 0,
            'constraint_enforcements': 0,
            'attractor_preservations': 0,
            'total_steps': 0
        }
        for i in range(1, n_steps):
            states[i] = self.step(t[i-1], states[i-1], dt)
        solve_time = self.performance_monitor.stop_timer("solve_total")
        logger.info(f"求解完成，耗时: {solve_time:.6f}秒，步数: {n_steps}")
        return t, states

    def get_performance_report(self):
        return self.performance_monitor.get_performance_report()

    def plot_performance(self, output_dir="performance_plots"):
        self.performance_monitor.plot_performance(output_dir)

class OptimizedParameterSpaceExplorer:
    def __init__(self, base_params=None, output_dir="parameter_exploration"):
        self.base_params = base_params or {"sigma": 10.0, "rho": 28.0, "beta": 8/3}
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.performance_monitor = PerformanceMonitor()

    def generate_parameter_grid(self, param_ranges, n_points=10):
        param_values = {}
        for param, (min_val, max_val) in param_ranges.items():
            param_values[param] = np.linspace(min_val, max_val, n_points)
        sigma_values = param_values.get("sigma", [self.base_params["sigma"]])
        rho_values = param_values.get("rho", [self.base_params["rho"]])
        beta_values = param_values.get("beta", [self.base_params["beta"]])
        param_sets = []
        for sigma in sigma_values:
            for rho in rho_values:
                for beta in beta_values:
                    param_sets.append((sigma, rho, beta))
        return param_sets

    def explore_parameter_space(self, param_ranges, t_span, initial_state, dt, n_points=10, n_workers=None):
        self.performance_monitor.start_timer("parameter_exploration")
        param_sets = self.generate_parameter_grid(param_ranges, n_points)
        logger.info(f"生成了{len(param_sets)}个参数集")
        results = solve_batch_parallel(param_sets, t_span, initial_state, dt, n_workers)
        exploration_time = self.performance_monitor.stop_timer("parameter_exploration")
        logger.info(f"参数空间探索完成，耗时: {exploration_time:.6f}秒，参数集数量: {len(param_sets)}")
        return results, param_sets

    def analyze_results(self, results, param_sets, analysis_type="stability"):
        self.performance_monitor.start_timer(f"analyze_{analysis_type}")
        analysis_results = {}
        if analysis_type == "stability":
            for idx, (t, states) in results.items():
                sigma, rho, beta = param_sets[idx]
                has_nan = np.any(np.isnan(states))
                has_inf = np.any(np.isinf(states))
                n_points = len(states)
                last_portion = states[int(0.9 * n_points):]
                max_value = np.max(np.abs(last_portion))
                is_divergent = max_value > 1000
                analysis_results[idx] = {
                    "params": (sigma, rho, beta),
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "is_divergent": is_divergent,
                    "max_value": max_value
                }
        elif analysis_type == "lyapunov":
            for idx, (t, states) in results.items():
                sigma, rho, beta = param_sets[idx]
                diffs = np.diff(states, axis=0)
                norms = np.linalg.norm(diffs, axis=1)
                valid_norms = norms[norms > 1e-10]
                if len(valid_norms) > 1:
                    growth_rates = np.log(valid_norms[1:] / valid_norms[:-1])
                    lyapunov_estimate = np.mean(growth_rates) / (t[1] - t[0])
                else:
                    lyapunov_estimate = 0
                analysis_results[idx] = {
                    "params": (sigma, rho, beta),
                    "lyapunov_estimate": lyapunov_estimate
                }
        elif analysis_type == "energy":
            for idx, (t, states) in results.items():
                sigma, rho, beta = param_sets[idx]
                energies = np.zeros(len(states))
                for i, state in enumerate(states):
                    x, y, z = state
                    energies[i] = 0.5 * (x**2 + y**2 + (z - rho)**2)
                mean_energy = np.mean(energies)
                std_energy = np.std(energies)
                min_energy = np.min(energies)
                max_energy = np.max(energies)
                analysis_results[idx] = {
                    "params": (sigma, rho, beta),
                    "mean_energy": mean_energy,
                    "std_energy": std_energy,
                    "min_energy": min_energy,
                    "max_energy": max_energy,
                    "energy_range": max_energy - min_energy
                }
        analysis_time = self.performance_monitor.stop_timer(f"analyze_{analysis_type}")
        logger.info(f"{analysis_type}分析完成，耗时: {analysis_time:.6f}秒")
        return analysis_results

    def visualize_parameter_space(self, analysis_results, param_ranges, analysis_type="stability"):
        self.performance_monitor.start_timer(f"visualize_{analysis_type}")
        params_list = [result["params"] for result in analysis_results.values()]
        if analysis_type == "stability":
            values = [1 if result["is_divergent"] else 0 for result in analysis_results.values()]
            title = "参数空间稳定性分析"
            cmap = "RdYlGn_r"
            label = "是否发散"
        elif analysis_type == "lyapunov":
            values = [result["lyapunov_estimate"] for result in analysis_results.values()]
            title = "参数空间Lyapunov指数估计"
            cmap = "plasma"
            label = "Lyapunov指数估计"
        elif analysis_type == "energy":
            values = [result["energy_range"] for result in analysis_results.values()]
            title = "参数空间能量范围分析"
            cmap = "viridis"
            label = "能量范围"
        unique_params = {}
        for param in ["sigma", "rho", "beta"]:
            if param in param_ranges:
                unique_params[param] = sorted(set([p[["sigma", "rho", "beta"].index(param)] for p in params_list]))
        n_dims = len(unique_params)
        if n_dims == 1:
            param = list(unique_params.keys())[0]
            param_values = unique_params[param]
            plt.figure(figsize=(10, 6))
            plt.plot(param_values, values, 'o-')
            plt.xlabel(param)
            plt.ylabel(label)
            plt.title(f"{title} - {param}变化")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, f"{analysis_type}_{param}.png"), dpi=300)
            plt.close()
        elif n_dims == 2:
            params = list(unique_params.keys())
            param1, param2 = params
            param1_values = unique_params[param1]
            param2_values = unique_params[param2]
            param1_idx = ["sigma", "rho", "beta"].index(param1)
            param2_idx = ["sigma", "rho", "beta"].index(param2)
            grid_values = np.zeros((len(param1_values), len(param2_values)))
            for idx, result in analysis_results.items():
                p = result["params"]
                i = param1_values.index(p[param1_idx])
                j = param2_values.index(p[param2_idx])
                if analysis_type == "stability":
                    grid_values[i, j] = 1 if result["is_divergent"] else 0
                elif analysis_type == "lyapunov":
                    grid_values[i, j] = result["lyapunov_estimate"]
                elif analysis_type == "energy":
                    grid_values[i, j] = result["energy_range"]
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(param2_values, param1_values, grid_values, cmap=cmap, shading='auto')
            plt.colorbar(label=label)
            plt.xlabel(param2)
            plt.ylabel(param1)
            plt.title(f"{title} - {param1}和{param2}变化")
            plt.savefig(os.path.join(self.output_dir, f"{analysis_type}_{param1}_{param2}.png"), dpi=300)
            plt.close()
        elif n_dims == 3:
            params = ["sigma", "rho", "beta"]
            for i in range(3):
                for j in range(i+1, 3):
                    param1, param2 = params[i], params[j]
                    param3 = [p for p in params if p != param1 and p != param2][0]
                    param3_base_value = self.base_params[param3]
                    param3_values = unique_params[param3]
                    param3_base_idx = min(range(len(param3_values)),
                                         key=lambda i: abs(param3_values[i] - param3_base_value))
                    param3_actual_value = param3_values[param3_base_idx]
                    param1_values = unique_params[param1]
                    param2_values = unique_params[param2]
                    param1_idx = params.index(param1)
                    param2_idx = params.index(param2)
                    param3_idx = params.index(param3)
                    grid_values = np.zeros((len(param1_values), len(param2_values)))
                    grid_mask = np.ones((len(param1_values), len(param2_values)), dtype=bool)
                    for idx, result in analysis_results.items():
                        p = result["params"]
                        if abs(p[param3_idx] - param3_actual_value) < 1e-6:
                            i = param1_values.index(p[param1_idx])
                            j = param2_values.index(p[param2_idx])
                            if analysis_type == "stability":
                                grid_values[i, j] = 1 if result["is_divergent"] else 0
                            elif analysis_type == "lyapunov":
                                grid_values[i, j] = result["lyapunov_estimate"]
                            elif analysis_type == "energy":
                                grid_values[i, j] = result["energy_range"]
                            grid_mask[i, j] = False
                    masked_grid = np.ma.array(grid_values, mask=grid_mask)
                    plt.figure(figsize=(10, 8))
                    plt.pcolormesh(param2_values, param1_values, masked_grid, cmap=cmap, shading='auto')
                    plt.colorbar(label=label)
                    plt.xlabel(param2)
                    plt.ylabel(param1)
                    plt.title(f"{title} - {param1}和{param2}变化 ({param3}={param3_actual_value:.2f})")
                    plt.savefig(os.path.join(self.output_dir,
                                            f"{analysis_type}_{param1}_{param2}_{param3}{param3_actual_value:.2f}.png"),
                               dpi=300)
                    plt.close()
        visualization_time = self.performance_monitor.stop_timer(f"visualize_{analysis_type}")
        logger.info(f"{analysis_type}可视化完成，耗时: {visualization_time:.6f}秒")

    def get_performance_report(self):
        return self.performance_monitor.get_performance_report()

    def plot_performance(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "performance")
        self.performance_monitor.plot_performance(output_dir)

if __name__ == "__main__":
    from lorenz_system import LorenzSystem
    system = LorenzSystem()
    integrator = OptimizedStructurePreservingIntegrator(system)
    t_span = [0, 100]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    t, states = integrator.solve(t_span, initial_state, dt)
    print(integrator.get_performance_report())
    integrator.plot_performance()
    explorer = OptimizedParameterSpaceExplorer()
    param_ranges = {
        "sigma": (8, 12),
        "rho": (20, 35),
        "beta": (2, 4)
    }
    results, param_sets = explorer.explore_parameter_space(
        param_ranges, [0, 50], [1.0, 1.0, 1.0], 0.01, n_points=5
    )
    stability_results = explorer.analyze_results(results, param_sets, "stability")
    lyapunov_results = explorer.analyze_results(results, param_sets, "lyapunov")
    energy_results = explorer.analyze_results(results, param_sets, "energy")
    explorer.visualize_parameter_space(stability_results, param_ranges, "stability")
    explorer.visualize_parameter_space(lyapunov_results, param_ranges, "lyapunov")
    explorer.visualize_parameter_space(energy_results, param_ranges, "energy")
    print(explorer.get_performance_report())
