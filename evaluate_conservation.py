import numpy as np
import matplotlib.pyplot as plt
import pickle
from lorenz_system import LorenzSystem
from numerical_solvers import (
    EulerSolver, MidpointSolver, RK2Solver, RK4Solver,
    SymplecticEulerSolver, EnergyPreservingSolver
)

def load_reference_solution(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['t'], data['states']

def calculate_volume_contraction(solver, t_span, initial_state, dt):
    t, states = solver.solve(t_span, initial_state, dt)
    
    lorenz = LorenzSystem()
    theoretical_rate = lorenz.volume_contraction_rate
    
    volume_contraction = np.zeros(len(t)-3)
    t_reduced = t[2:-1]
    
    initial_volume = None
    
    for i in range(len(t)-3):
        p1 = states[i]
        p2 = states[i+1]
        p3 = states[i+2]
        p4 = states[i+3]
        
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1
        
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        
        if volume < 1e-10:
            volume = 1e-10
        
        if i == 0:
            initial_volume = volume
        
        if initial_volume is not None and initial_volume > 1e-10 and t[i+3] - t[0] > 1e-10 :
             volume_contraction[i] = np.log(volume / initial_volume) / (t[i+3] - t[0])
        else:
             volume_contraction[i] = 0


    return t_reduced, volume_contraction, theoretical_rate


def calculate_pseudo_energy(solver, t_span, initial_state, dt):
    t, states = solver.solve(t_span, initial_state, dt)
    lorenz = LorenzSystem()
    pseudo_energy = np.array([lorenz.pseudo_energy(state) for state in states])
    return t, pseudo_energy

def analyze_attractor_structure(solver, t_span, initial_state, dt):
    t, states = solver.solve(t_span, initial_state, dt)
    
    x_range = (np.min(states[:, 0]), np.max(states[:, 0]))
    y_range = (np.min(states[:, 1]), np.max(states[:, 1]))
    z_range = (np.min(states[:, 2]), np.max(states[:, 2]))
    
    center = np.mean(states, axis=0)
    
    distances = np.sqrt(np.sum((states - center) ** 2, axis=1))
    radius = np.max(distances)
    
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])
    
    return {
        't': t,
        'states': states,
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'center': center,
        'radius': radius,
        'volume': volume
    }

def plot_volume_contraction(results_dict, title="Volume Contraction Rate"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    theoretical_rate = results_dict[list(results_dict.keys())[0]]['theoretical_rate']
    ax.axhline(y=theoretical_rate, color='k', linestyle='--',
               label=f'Theoretical Rate: {theoretical_rate:.4f}')
    
    for solver_name, results in results_dict.items():
        ax.plot(results['t'], results['volume_contraction'], label=solver_name)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume Contraction Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.savefig('volume_contraction_comparison.png', dpi=300)
    plt.close()

def plot_pseudo_energy(results_dict, title="Pseudo-Energy Function"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for solver_name, results in results_dict.items():
        ax.plot(results['t'], results['pseudo_energy'], label=solver_name)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Pseudo-Energy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.savefig('pseudo_energy_comparison.png', dpi=300)
    plt.close()

def plot_attractor_statistics(results_dict, title="Attractor Structure Statistics"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    solver_names = list(results_dict.keys())
    x_ranges = [results_dict[name]['x_range'] for name in solver_names]
    y_ranges = [results_dict[name]['y_range'] for name in solver_names]
    z_ranges = [results_dict[name]['z_range'] for name in solver_names]
    volumes = [results_dict[name]['volume'] for name in solver_names]
    radii = [results_dict[name]['radius'] for name in solver_names]
    
    axs[0, 0].bar(solver_names, [x[1] - x[0] for x in x_ranges])
    axs[0, 0].set_title('X Range')
    axs[0, 0].set_ylabel('Range')
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    axs[0, 1].bar(solver_names, [y[1] - y[0] for y in y_ranges])
    axs[0, 1].set_title('Y Range')
    axs[0, 1].set_ylabel('Range')
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    axs[1, 0].bar(solver_names, [z[1] - z[0] for z in z_ranges])
    axs[1, 0].set_title('Z Range')
    axs[1, 0].set_ylabel('Range')
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    axs[1, 1].bar(solver_names, volumes)
    axs[1, 1].set_title('Attractor Volume')
    axs[1, 1].set_ylabel('Volume')
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('attractor_statistics.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(solver_names, radii)
    plt.title('Attractor Radius')
    plt.ylabel('Radius')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('attractor_radius.png', dpi=300)
    plt.close()

def create_conservation_table(volume_results, energy_results, attractor_results):
    headers = ["Solver", "Volume Rate Error", "Energy Stability", "Attractor Volume", "Attractor Radius"]
    
    rows = []
    for solver_name in volume_results.keys():
        theoretical_rate = volume_results[solver_name]['theoretical_rate']
        volume_rate = volume_results[solver_name]['volume_contraction']
        volume_error = np.mean(np.abs(volume_rate - theoretical_rate))
        
        energy = energy_results[solver_name]['pseudo_energy']
        energy_stability = np.std(energy) / np.mean(energy) if np.mean(energy) != 0 else 0
        
        attractor_volume = attractor_results[solver_name]['volume']
        attractor_radius = attractor_results[solver_name]['radius']
        
        row = [
            solver_name,
            f"{volume_error:.6f}",
            f"{energy_stability:.6f}",
            f"{attractor_volume:.6f}",
            f"{attractor_radius:.6f}"
        ]
        rows.append(row)
    
    rows.sort(key=lambda x: float(x[1]))
    
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    
    return table

if __name__ == "__main__":
    print("评估各求解器的守恒性质...")
    
    t_span = [0, 50]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem()  # Create LorenzSystem instance outside the loop

    solvers = {
        "Euler": EulerSolver(lorenz),
        "Midpoint": MidpointSolver(lorenz),
        "RK2": RK2Solver(lorenz),
        "RK4": RK4Solver(lorenz),
        "Symplectic": SymplecticEulerSolver(lorenz),
        "Energy Preserving": EnergyPreservingSolver(lorenz)
    }
    
    print("评估体积收缩率...")
    volume_results = {}
    for name, solver in solvers.items():
        print(f"  评估 {name} 求解器的体积收缩率...")
        t, volume_contraction, theoretical_rate = calculate_volume_contraction(
            solver, t_span, initial_state, dt)
        volume_results[name] = {
            't': t,
            'volume_contraction': volume_contraction,
            'theoretical_rate': theoretical_rate
        }
    
    print("评估伪能量函数...")
    energy_results = {}
    for name, solver in solvers.items():
        print(f"  评估 {name} 求解器的伪能量函数...")
        t, pseudo_energy = calculate_pseudo_energy(solver, t_span, initial_state, dt)
        energy_results[name] = {
            't': t,
            'pseudo_energy': pseudo_energy
        }
    
    print("分析吸引子结构...")
    attractor_results = {}
    for name, solver in solvers.items():
        print(f"  分析 {name} 求解器的吸引子结构...")
        attractor_results[name] = analyze_attractor_structure(solver, t_span, initial_state, dt)
    
    print("绘制体积收缩率对比图...")
    plot_volume_contraction(volume_results)
    
    print("绘制伪能量函数对比图...")
    plot_pseudo_energy(energy_results)
    
    print("绘制吸引子结构统计对比图...")
    plot_attractor_statistics(attractor_results)
    
    print("创建守恒性质对比表格...")
    table = create_conservation_table(volume_results, energy_results, attractor_results)
    
    with open('conservation_properties.md', 'w') as f:
        f.write("# 求解器守恒性质对比\n\n")
        f.write("## 评估参数\n")
        f.write(f"- 时间范围: {t_span}\n")
        f.write(f"- 初始状态: {initial_state}\n")
        f.write(f"- 时间步长: {dt}\n\n")
        f.write("## 守恒性质对比表格\n\n")
        f.write(table)
        f.write("\n## 说明\n")
        f.write("- Volume Rate Error: 体积收缩率与理论值的平均绝对偏差（越小越好）\n")
        f.write("- Energy Stability: 伪能量的变异系数（标准差/平均值，越小越稳定）\n")
        f.write("- Attractor Volume: 吸引子的近似体积\n")
        f.write("- Attractor Radius: 吸引子的半径（到中心的最大距离）\n")
    
    print("评估完成，结果已保存到 'conservation_properties.md'")
