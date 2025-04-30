import numpy as np
import matplotlib.pyplot as plt
from lorenz_system import LorenzSystem
from numerical_solvers import (
    NumericalSolver, EulerSolver, MidpointSolver, RK2Solver, RK4Solver,
    SymplecticEulerSolver, EnergyPreservingSolver
)
import time

class HybridAdaptiveSolver(NumericalSolver):
    def __init__(self, system=None, efficient_solver=None, accurate_solver=None,
                 lyapunov_threshold=0.5, transition_width=0.2):
        super().__init__(system)
        self.name = "Hybrid Adaptive Solver"
        
        self.efficient_solver = efficient_solver if efficient_solver is not None else RK2Solver(system)
        self.accurate_solver = accurate_solver if accurate_solver is not None else RK4Solver(system)
        
        self.lyapunov_threshold = lyapunov_threshold
        self.transition_width = transition_width
        
        self.previous_states = []
        self.max_history = 5
        
        self.solver_usage = {
            'efficient': 0,
            'accurate': 0,
            'mixed': 0
        }
    
    def estimate_local_sensitivity(self, t, state):
        if len(self.previous_states) < self.max_history:
            return 0.0
        
        derivatives = []
        for i in range(1, len(self.previous_states)):
            prev_state = self.previous_states[i-1]
            curr_state = self.previous_states[i]
            deriv = np.linalg.norm(curr_state - prev_state)
            derivatives.append(deriv)
        
        second_derivatives = []
        for i in range(1, len(derivatives)):
            second_deriv = abs(derivatives[i] - derivatives[i-1])
            second_derivatives.append(second_deriv)
        
        sensitivity = np.mean(second_derivatives) if second_derivatives else 0.0
        
        x, y, z = state
        distance_to_z_axis = np.sqrt(x**2 + y**2)
        
        if distance_to_z_axis < 2.0 or np.linalg.norm(state) < 5.0:
            sensitivity += 0.5
            
        if (-12.7 < x < 13.3) and (-21.8 < y < 24.9) and (0.9 < z < 40.5):
            if abs(x) > 8.0 and abs(y) > 8.0:
                sensitivity += 0.3
            if 20.0 < z < 35.0:
                sensitivity += 0.2
        
        return sensitivity
    
    def get_mixing_ratio(self, sensitivity):
        if sensitivity <= self.lyapunov_threshold - self.transition_width/2:
            ratio = 0.0
        elif sensitivity >= self.lyapunov_threshold + self.transition_width/2:
            ratio = 1.0
        else:
            lower_bound = self.lyapunov_threshold - self.transition_width/2
            upper_bound = self.lyapunov_threshold + self.transition_width/2
            ratio = (sensitivity - lower_bound) / self.transition_width
        
        return ratio
    
    def step(self, t, state, dt):
        self.previous_states.append(state.copy())
        if len(self.previous_states) > self.max_history:
            self.previous_states.pop(0)
        
        sensitivity = self.estimate_local_sensitivity(t, state)
        ratio = self.get_mixing_ratio(sensitivity)
        
        if ratio == 0.0:
            next_state = self.efficient_solver.step(t, state, dt)
            self.solver_usage['efficient'] += 1
        elif ratio == 1.0:
            next_state = self.accurate_solver.step(t, state, dt)
            self.solver_usage['accurate'] += 1
        else:
            efficient_state = self.efficient_solver.step(t, state, dt)
            accurate_state = self.accurate_solver.step(t, state, dt)
            next_state = (1 - ratio) * efficient_state + ratio * accurate_state
            self.solver_usage['mixed'] += 1
        
        return next_state
    
    def get_usage_statistics(self):
        total_steps = sum(self.solver_usage.values())
        if total_steps == 0:
            return {k: 0.0 for k in self.solver_usage}
        
        return {k: v / total_steps for k, v in self.solver_usage.items()}


class AdaptiveStepSizeSolver(HybridAdaptiveSolver):
    def __init__(self, system=None, efficient_solver=None, accurate_solver=None,
                 lyapunov_threshold=0.5, transition_width=0.2,
                 min_dt_factor=0.1, max_dt_factor=2.0):
        super().__init__(system, efficient_solver, accurate_solver,
                         lyapunov_threshold, transition_width)
        self.name = "Adaptive Step Size Solver"
        
        self.min_dt_factor = min_dt_factor
        self.max_dt_factor = max_dt_factor
        
        self.dt_factors = []
    
    def solve(self, t_span, initial_state, dt):
        t_start, t_end = t_span
        t_current = t_start
        state_current = np.array(initial_state)
        
        t_points = [t_current]
        states = [state_current.copy()]
        
        self.previous_states = []
        self.solver_usage = {
            'efficient': 0,
            'accurate': 0,
            'mixed': 0
        }
        self.dt_factors = []
        
        while t_current < t_end:
            sensitivity = self.estimate_local_sensitivity(t_current, state_current)
            dt_factor = self.calculate_dt_factor(sensitivity)
            self.dt_factors.append(dt_factor)
            
            current_dt = dt * dt_factor
            
            if t_current + current_dt > t_end:
                current_dt = t_end - t_current
            
            if current_dt <= 1e-9: # Avoid excessively small steps
                 break

            state_current = self.step(t_current, state_current, current_dt)
            t_current += current_dt
            
            t_points.append(t_current)
            states.append(state_current.copy())
        
        return np.array(t_points), np.array(states)
    
    def calculate_dt_factor(self, sensitivity):
        if sensitivity <= self.lyapunov_threshold - self.transition_width:
            factor = self.max_dt_factor
        elif sensitivity >= self.lyapunov_threshold + self.transition_width:
            factor = self.min_dt_factor
        else:
            ratio = (sensitivity - (self.lyapunov_threshold - self.transition_width)) / (2 * self.transition_width)
            factor = self.max_dt_factor - ratio * (self.max_dt_factor - self.min_dt_factor)
            # Ensure factor is within bounds
            factor = max(self.min_dt_factor, min(self.max_dt_factor, factor))

        return factor


def test_hybrid_solver():
    print("测试混合自适应求解器...")
    
    t_span = [0, 50]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem()
    efficient_solver = RK2Solver(lorenz)
    accurate_solver = RK4Solver(lorenz)
    
    hybrid_solver = HybridAdaptiveSolver(
        lorenz, efficient_solver, accurate_solver,
        lyapunov_threshold=0.5, transition_width=0.2
    )
    
    t, states = hybrid_solver.solve(t_span, initial_state, dt)
    usage_stats = hybrid_solver.get_usage_statistics()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hybrid Adaptive Solver Trajectory')
    
    plt.savefig('hybrid_solver_trajectory.png', dpi=300)
    plt.close()
    
    print("求解器使用统计:")
    print(f"  高效求解器 (RK2): {usage_stats['efficient']*100:.2f}%")
    print(f"  高精度求解器 (RK4): {usage_stats['accurate']*100:.2f}%")
    print(f"  混合使用: {usage_stats['mixed']*100:.2f}%")
    
    return t, states, usage_stats


def test_adaptive_step_size_solver():
    print("测试自适应步长混合求解器...")
    
    t_span = [0, 50]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem()
    efficient_solver = RK2Solver(lorenz)
    accurate_solver = RK4Solver(lorenz)
    
    adaptive_solver = AdaptiveStepSizeSolver(
        lorenz, efficient_solver, accurate_solver,
        lyapunov_threshold=0.5, transition_width=0.2,
        min_dt_factor=0.2, max_dt_factor=2.0
    )
    
    t, states = adaptive_solver.solve(t_span, initial_state, dt)
    usage_stats = adaptive_solver.get_usage_statistics()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Adaptive Step Size Solver Trajectory')
    
    plt.savefig('adaptive_step_size_trajectory.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    if adaptive_solver.dt_factors: # Check if list is not empty
        plt.plot(adaptive_solver.dt_factors)
        plt.xlabel('Step')
        plt.ylabel('dt Factor')
        plt.title('Step Size Adaptation')
        plt.grid(True)
        plt.savefig('step_size_adaptation.png', dpi=300)
    plt.close()
    
    print("求解器使用统计:")
    print(f"  高效求解器 (RK2): {usage_stats['efficient']*100:.2f}%")
    print(f"  高精度求解器 (RK4): {usage_stats['accurate']*100:.2f}%")
    print(f"  混合使用: {usage_stats['mixed']*100:.2f}%")
    if adaptive_solver.dt_factors: # Check if list is not empty
        print(f"  平均步长因子: {np.mean(adaptive_solver.dt_factors):.4f}")
        print(f"  最小步长因子: {np.min(adaptive_solver.dt_factors):.4f}")
        print(f"  最大步长因子: {np.max(adaptive_solver.dt_factors):.4f}")
    else:
        print("  无步长因子统计信息 (可能没有执行任何步骤)")

    
    return t, states, usage_stats, adaptive_solver.dt_factors


def compare_solvers():
    print("比较不同求解器的性能...")
    
    t_span = [0, 50]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem()
    solvers = {
        "RK2": RK2Solver(lorenz),
        "RK4": RK4Solver(lorenz),
        "Hybrid": HybridAdaptiveSolver(
            lorenz, RK2Solver(lorenz), RK4Solver(lorenz),
            lyapunov_threshold=0.5, transition_width=0.2
        ),
        "Adaptive": AdaptiveStepSizeSolver(
            lorenz, RK2Solver(lorenz), RK4Solver(lorenz),
            lyapunov_threshold=0.5, transition_width=0.2,
            min_dt_factor=0.2, max_dt_factor=2.0
        )
    }
    
    results = {}
    
    for name, solver in solvers.items():
        print(f"  求解 {name}...")
        start_time = time.time()
        t, states = solver.solve(t_span, initial_state, dt)
        end_time = time.time()
        
        results[name] = {
            't': t,
            'states': states,
            'time': end_time - start_time
        }
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'g', 'm']
    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['states'][:, 0], result['states'][:, 1], result['states'][:, 2],
                colors[i], linewidth=0.5, label=name)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Solver Comparison')
    ax.legend()
    
    plt.savefig('solver_comparison.png', dpi=300)
    plt.close()
    
    with open('hybrid_solver_performance.md', 'w') as f:
        f.write("# 混合自适应求解器性能对比\n\n")
        f.write("## 评估参数\n")
        f.write(f"- 时间范围: {t_span}\n")
        f.write(f"- 初始状态: {initial_state}\n")
        f.write(f"- 基准时间步长: {dt}\n\n")
        f.write("## 性能对比表格\n\n")
        f.write("| Solver | Computation Time (s) | Steps | Avg Step Size |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        for name, result in results.items():
             steps = len(result['t']) -1 # Number of steps is len(t) - 1
             avg_step = (t_span[1] - t_span[0]) / steps if steps > 0 else 0
             f.write(f"| {name} | {result['time']:.6f} | {steps} | {avg_step:.6f} |\n")
        
        if isinstance(solvers["Hybrid"], HybridAdaptiveSolver):
            f.write("\n## 混合求解器使用统计\n\n")
            usage_stats = solvers["Hybrid"].get_usage_statistics()
            f.write(f"- 高效求解器 (RK2): {usage_stats['efficient']*100:.2f}%\n")
            f.write(f"- 高精度求解器 (RK4): {usage_stats['accurate']*100:.2f}%\n")
            f.write(f"- 混合使用: {usage_stats['mixed']*100:.2f}%\n")
        
        if isinstance(solvers["Adaptive"], AdaptiveStepSizeSolver):
             f.write("\n## 自适应步长求解器统计\n\n")
             usage_stats = solvers["Adaptive"].get_usage_statistics()
             dt_factors = solvers["Adaptive"].dt_factors
             f.write(f"- 高效求解器 (RK2): {usage_stats['efficient']*100:.2f}%\n")
             f.write(f"- 高精度求解器 (RK4): {usage_stats['accurate']*100:.2f}%\n")
             f.write(f"- 混合使用: {usage_stats['mixed']*100:.2f}%\n")
             if dt_factors: # Check if list is not empty
                 f.write(f"- 平均步长因子: {np.mean(dt_factors):.4f}\n")
                 f.write(f"- 最小步长因子: {np.min(dt_factors):.4f}\n")
                 f.write(f"- 最大步长因子: {np.max(dt_factors):.4f}\n")
             else:
                 f.write("- 无步长因子统计信息\n")

    print("性能对比完成，结果已保存到 'hybrid_solver_performance.md'")
    
    return results


if __name__ == "__main__":
    hybrid_t, hybrid_states, hybrid_usage = test_hybrid_solver()
    adaptive_t, adaptive_states, adaptive_usage, dt_factors = test_adaptive_step_size_solver()
    comparison_results = compare_solvers()
