import numpy as np
import matplotlib.pyplot as plt
import pickle
from lorenz_system import LorenzSystem
from numerical_solvers import (
    EulerSolver, MidpointSolver, RK2Solver, RK4Solver,
    SymplecticEulerSolver, EnergyPreservingSolver
)
from hybrid_solver import HybridAdaptiveSolver
from structure_preserving_integrator import StructurePreservingIntegrator
import os

def calculate_error_metrics(reference_states, solver_states):
    min_length = min(len(reference_states), len(solver_states))
    reference_states = reference_states[:min_length]
    solver_states = solver_states[:min_length]
    
    euclidean_errors = np.sqrt(np.sum((solver_states - reference_states)**2, axis=1))
    
    norms = np.sqrt(np.sum(reference_states**2, axis=1))
    norms[norms == 0] = 1e-10
    relative_errors = euclidean_errors / norms
    
    error_metrics = {
        'mean_euclidean': np.mean(euclidean_errors),
        'max_euclidean': np.max(euclidean_errors),
        'mean_relative': np.mean(relative_errors),
        'max_relative': np.max(relative_errors)
    }
    
    return error_metrics

def plot_optimized_error_convergence():
    print("生成优化后的误差收敛性图表...")
    
    t_span = [0, 10]
    initial_state = [1.0, 1.0, 1.0]
    
    lorenz = LorenzSystem()
    
    reference_dt = 0.001
    reference_solver = RK4Solver(lorenz)
    reference_t, reference_states = reference_solver.solve(t_span, initial_state, reference_dt)
    
    solver_names_map = {
        "Euler": "欧拉法",
        "Midpoint": "中点法",
        "RK2": "RK2方法",
        "RK4": "RK4方法",
        "Symplectic": "辛方法",
        "Energy": "能量守恒方法"
    }
    
    solvers = {
        "Euler": EulerSolver(lorenz),
        "Midpoint": MidpointSolver(lorenz),
        "RK2": RK2Solver(lorenz),
        "RK4": RK4Solver(lorenz),
        "Symplectic": SymplecticEulerSolver(lorenz),
        "Energy": EnergyPreservingSolver(lorenz)
    }
    
    colors = {
        "Euler": 'blue',
        "Midpoint": 'green',
        "RK2": 'red',
        "RK4": 'purple',
        "Symplectic": 'orange',
        "Energy": 'brown'
    }
    
    markers = {
        "Euler": 'o',
        "Midpoint": 's',
        "RK2": '^',
        "RK4": 'D',
        "Symplectic": 'p',
        "Energy": '*'
    }
    
    dt_values = np.logspace(-3, -1, 10)
    
    convergence_results = {}
    
    for solver_name, solver in solvers.items():
        convergence_results[solver_name] = {
            'mean_errors': [],
            'max_errors': []
        }
        
        for dt in dt_values:
            try:
                t, states = solver.solve(t_span, initial_state, dt)
                error_metrics = calculate_error_metrics(reference_states, states)
                convergence_results[solver_name]['mean_errors'].append(error_metrics['mean_euclidean'])
                convergence_results[solver_name]['max_errors'].append(error_metrics['max_euclidean'])
            except:
                convergence_results[solver_name]['mean_errors'].append(np.nan)
                convergence_results[solver_name]['max_errors'].append(np.nan)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    for solver_name in solvers.keys():
        display_name = solver_names_map.get(solver_name, solver_name)
        axs[0].loglog(dt_values, convergence_results[solver_name]['mean_errors'],
                     marker=markers[solver_name], color=colors[solver_name],
                     linestyle='-', linewidth=2, markersize=8,
                     label=display_name)
    
    axs[0].set_xlabel('时间步长 (dt)', fontsize=14)
    axs[0].set_ylabel('平均欧几里得误差', fontsize=14)
    axs[0].set_title('平均误差收敛性', fontsize=16)
    axs[0].grid(True, which="both", ls="-", alpha=0.2)
    axs[0].legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    
    for solver_name in solvers.keys():
        display_name = solver_names_map.get(solver_name, solver_name)
        axs[1].loglog(dt_values, convergence_results[solver_name]['max_errors'],
                     marker=markers[solver_name], color=colors[solver_name],
                     linestyle='-', linewidth=2, markersize=8,
                     label=display_name)
    
    axs[1].set_xlabel('时间步长 (dt)', fontsize=14)
    axs[1].set_ylabel('最大欧几里得误差', fontsize=14)
    axs[1].set_title('最大误差收敛性', fontsize=16)
    axs[1].grid(True, which="both", ls="-", alpha=0.2)
    axs[1].legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=True)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('不同求解器的误差收敛性对比', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_dir = "error_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'error_convergence_optimized.png'), dpi=300, bbox_inches='tight')
    print(f"优化后的误差收敛性图表已保存到 '{output_dir}/error_convergence_optimized.png'")
    plt.close()

if __name__ == "__main__":
    plot_optimized_error_convergence()
