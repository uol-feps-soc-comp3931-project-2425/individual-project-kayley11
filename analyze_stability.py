
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import solve_ivp
from lorenz_system import LorenzSystem
from numerical_solvers import (
    EulerSolver, MidpointSolver, RK2Solver, RK4Solver, 
    SymplecticEulerSolver, EnergyPreservingSolver
)

def load_reference_solution(filename):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['t'], data['states']

def calculate_local_lyapunov_exponent(solver, t_span, initial_state, dt, perturbation_size=1e-10, window_size=20):
    """
    计算局部Lyapunov指数
    
    参数:
    solver: 求解器对象
    t_span: 时间范围
    initial_state: 初始状态
    dt: 时间步长
    perturbation_size: 初始扰动大小
    window_size: 计算局部指数的窗口大小（时间点数）
    
    返回:
    t: 时间点
    local_lyapunov: 局部Lyapunov指数
    """
   
    t, states = solver.solve(t_span, initial_state, dt)
    
    
    n_steps = len(t)
    local_lyapunov = np.zeros(n_steps - window_size)
    t_reduced = t[window_size:]  # 减少时间点以匹配局部指数数组长度
    
    
    for i in range(n_steps - window_size):
       
        perturbed_state = states[i].copy()
        
        perturbation = np.random.randn(3) * perturbation_size
        perturbed_state += perturbation
        
        
        _, original_window = solver.solve([t[i], t[i+window_size]], states[i], dt)
        _, perturbed_window = solver.solve([t[i], t[i+window_size]], perturbed_state, dt)
        
        
        initial_distance = np.linalg.norm(perturbation)
        final_distance = np.linalg.norm(perturbed_window[-1] - original_window[-1])
        
        
        if final_distance < 1e-15 or initial_distance < 1e-15:
            local_lyapunov[i] = 0
        else:
            
            time_diff = t[i+window_size] - t[i]
            local_lyapunov[i] = (1 / time_diff) * np.log(final_distance / initial_distance)
    
    return t_reduced, local_lyapunov

def test_step_size_stability(solver_class, t_span, initial_state, dt_values):
    
    stability_metrics = []
    
    for dt in dt_values:
        try:
            
            solver = solver_class()
            
            
            t, states = solver.solve(t_span, initial_state, dt)
            
           
            if np.any(np.isnan(states)) or np.any(np.isinf(states)):
                
                stability_metrics.append(float('inf'))
            else:
                
                max_value = np.max(np.abs(states))
                stability_metrics.append(max_value)
        except Exception as e:
            
            print(f"  步长 {dt} 求解失败: {str(e)}")
            stability_metrics.append(float('inf'))
    
    return dt_values, stability_metrics

def evaluate_long_term_physical_consistency(solver, t_span, initial_state, dt):

    
    t, states = solver.solve(t_span, initial_state, dt)
    
   
    lorenz = LorenzSystem()
    
    
    x_range = (np.min(states[:, 0]), np.max(states[:, 0]))
    y_range = (np.min(states[:, 1]), np.max(states[:, 1]))
    z_range = (np.min(states[:, 2]), np.max(states[:, 2]))
    
    
    z_always_positive = np.all(states[:, 2] > 0)
    
    
    pseudo_energies = np.array([lorenz.pseudo_energy(state) for state in states])
    energy_mean = np.mean(pseudo_energies)
    energy_std = np.std(pseudo_energies)
    energy_variation = energy_std / energy_mean
    
   
    has_instability = np.any(np.isnan(states)) or np.any(np.isinf(states))
    
    return {
        't': t,
        'states': states,
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'z_always_positive': z_always_positive,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'energy_variation': energy_variation,
        'has_instability': has_instability
    }

def plot_local_lyapunov(results_dict, title="Local Lyapunov Exponents"):

    fig, ax = plt.subplots(figsize=(12, 8))
    

    for solver_name, results in results_dict.items():
        ax.plot(results['t'], results['local_lyapunov'], label=solver_name)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Local Lyapunov Exponent')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # 添加零线
    ax.axhline(y=0, color='k', linestyle='--')
    
    plt.savefig('local_lyapunov_comparison.png', dpi=300)
    plt.close()
    
   
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for solver_name, results in results_dict.items():
        ax.hist(results['local_lyapunov'], bins=30, alpha=0.5, label=solver_name)
    
    ax.set_xlabel('Local Lyapunov Exponent')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Local Lyapunov Exponents')
    ax.legend()
    
    plt.savefig('local_lyapunov_histogram.png', dpi=300)
    plt.close()

def plot_step_size_stability(results_dict, title="Step Size Stability"):

    fig, ax = plt.subplots(figsize=(12, 8))
    
   
    for solver_name, results in results_dict.items():
        
        stability = np.array(results['stability'])
        stability[np.isinf(stability)] = 1e5
        
        ax.semilogy(results['dt_values'], stability, 'o-', label=solver_name)
    
    ax.set_xlabel('Time Step Size (dt)')
    ax.set_ylabel('Stability Metric (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.savefig('step_size_stability.png', dpi=300)
    plt.close()

def create_stability_table(lyapunov_results, step_size_results, physical_results):
   
    headers = ["Solver", "Max Lyapunov", "Critical dt", "Energy Variation", "Z Always Positive", "Instability"]
    
    rows = []
    for solver_name in lyapunov_results.keys():
        
        max_lyapunov = np.max(lyapunov_results[solver_name]['local_lyapunov'])
        
        
        dt_values = step_size_results[solver_name]['dt_values']
        stability = step_size_results[solver_name]['stability']
        critical_dt = 0
        for i, stab in enumerate(stability):
            if np.isinf(stab):
                if i > 0:
                    critical_dt = dt_values[i-1]
                break
            if i == len(stability) - 1:
                critical_dt = dt_values[i]
        
        
        energy_variation = physical_results[solver_name]['energy_variation']
        z_always_positive = "是" if physical_results[solver_name]['z_always_positive'] else "否"
        has_instability = "是" if physical_results[solver_name]['has_instability'] else "否"
        
        row = [
            solver_name,
            f"{max_lyapunov:.6f}",
            f"{critical_dt:.6f}",
            f"{energy_variation:.6f}",
            z_always_positive,
            has_instability
        ]
        rows.append(row)
    
   
    rows.sort(key=lambda x: float(x[1]))
    
   
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    
    return table

def identify_sensitive_regions(t, local_lyapunov, states, threshold=0.5):

   
    sensitive_indices = np.where(local_lyapunov > threshold)[0]
    
    if len(sensitive_indices) == 0:
        return [], []
    
    sensitive_states = states[sensitive_indices]
    sensitive_lyapunov = local_lyapunov[sensitive_indices]
    
    return sensitive_states, sensitive_lyapunov

def plot_sensitive_regions(states, sensitive_states, title="Sensitive Regions in 
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    

    ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.5, alpha=0.3, label='Trajectory')
    
 
    if len(sensitive_states) > 0:
        ax.scatter(sensitive_states[:, 0], sensitive_states[:, 1], sensitive_states[:, 2], 
                  c='r', s=20, label='Sensitive Regions')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.savefig('sensitive_regions.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("分析Lorenz系统的稳定性和Lyapunov指数...")
    
   
    t_span_lyapunov = [0, 50]  # 用于计算Lyapunov指数的时间范围
    t_span_physical = [0, 100]  # 用于评估长时间物理合理性的时间范围
    initial_state = [1.0, 1.0, 1.0]  # 初始状态
    dt = 0.01  # 基准时间步长
    
   
    solvers = {
        "Euler": EulerSolver(),
        "Midpoint": MidpointSolver(),
        "RK2": RK2Solver(),
        "RK4": RK4Solver(),
        "Symplectic": SymplecticEulerSolver(),
        "Energy Preserving": EnergyPreservingSolver()
    }
    
    # 计算局部Lyapunov指数
    print("计算局部Lyapunov指数...")
    lyapunov_results = {}
    for name, solver in solvers.items():
        print(f"  计算 {name} 求解器的局部Lyapunov指数...")
        t, local_lyapunov = calculate_local_lyapunov_exponent(
            solver, t_span_lyapunov, initial_state, dt)
        lyapunov_results[name] = {
            't': t,
            'local_lyapunov': local_lyapunov
        }
    
    # 测试不同步长下的数值稳定性
    print("测试不同步长下的数值稳定性...")
    dt_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    step_size_results = {}
    for name, solver_class in {
        "Euler": EulerSolver,
        "Midpoint": MidpointSolver,
        "RK2": RK2Solver,
        "RK4": RK4Solver,
        "Symplectic": SymplecticEulerSolver,
        "Energy Preserving": EnergyPreservingSolver
    }.items():
        print(f"  测试 {name} 求解器在不同步长下的稳定性...")
        dt_vals, stability = test_step_size_stability(
            solver_class, [0, 10], initial_state, dt_values)
        step_size_results[name] = {
            'dt_values': dt_vals,
            'stability': stability
        }
    
    # 评估长时间积分下解的物理合理性
    print("评估长时间积分下解的物理合理性...")
    physical_results = {}
    for name, solver in solvers.items():
        print(f"  评估 {name} 求解器的长时间物理合理性...")
        physical_results[name] = evaluate_long_term_physical_consistency(
            solver, t_span_physical, initial_state, dt)
    
    # 绘制局部Lyapunov指数对比图
    print("绘制局部Lyapunov指数对比图...")
    plot_local_lyapunov(lyapunov_results)
    
    # 绘制步长稳定性对比图
    print("绘制步长稳定性对比图...")
    plot_step_size_stability(step_size_results)
    
    # 创建稳定性和Lyapunov指数对比表格
    print("创建稳定性和Lyapunov指数对比表格...")
    table = create_stability_table(lyapunov_results, step_size_results, physical_results)
    
    # 保存表格到文件
    with open('stability_analysis.md', 'w') as f:
        f.write("# 求解器稳定性和Lyapunov指数分析\n\n")
        f.write("## 评估参数\n")
        f.write(f"- Lyapunov指数计算时间范围: {t_span_lyapunov}\n")
        f.write(f"- 物理合理性评估时间范围: {t_span_physical}\n")
        f.write(f"- 初始状态: {initial_state}\n")
        f.write(f"- 基准时间步长: {dt}\n\n")
        f.write("## 稳定性和Lyapunov指数对比表格\n\n")
        f.write(table)
        f.write("\n## 说明\n")
        f.write("- Max Lyapunov: 最大局部Lyapunov指数，值越大表示系统越敏感\n")
        f.write("- Critical dt: 临界时间步长，超过此值求解器会变得不稳定\n")
        f.write("- Energy Variation: 伪能量的变异系数（标准差/平均值），越小越稳定\n")
        f.write("- Z Always Positive: Lorenz系统中z坐标是否始终为正\n")
        f.write("- Instability: 是否出现数值不稳定性（NaN或无穷大）\n")
    
   
    print("识别Lorenz系统中的敏感区域...")
    rk4_t = lyapunov_results["RK4"]['t']
    rk4_lyapunov = lyapunov_results["RK4"]['local_lyapunov']
    rk4_states = physical_results["RK4"]['states'][:len(rk4_t)]  # 确保长度匹配
    
    
    sensitive_states, sensitive_lyapunov = identify_sensitive_regions(
        rk4_t, rk4_lyapunov, rk4_states, threshold=0.5)
    
    
    if len(sensitive_states) > 0:
        print(f"找到 {len(sensitive_states)} 个敏感区域点，绘制敏感区域图...")
        plot_sensitive_regions(rk4_states, sensitive_states)
        
        with open('sensitive_regions.md', 'w') as f:
            f.write("# Lorenz系统敏感区域分析\n\n")
            f.write("## 敏感区域统计\n")
            f.write(f"- 总轨迹点数: {len(rk4_states)}\n")
            f.write(f"- 敏感区域点数: {len(sensitive_states)}\n")
            f.write(f"- 敏感区域占比: {len(sensitive_states)/len(rk4_states)*100:.2f}%\n\n")
            f.write("## 敏感区域特征\n")
            f.write(f"- 平均Lyapunov指数: {np.mean(sensitive_lyapunov):.6f}\n")
            f.write(f"- 最大Lyapunov指数: {np.max(sensitive_lyapunov):.6f}\n")
            f.write(f"- X坐标范围: [{np.min(sensitive_states[:, 0]):.4f}, {np.max(sensitive_states[:, 0]):.4f}]\n")
            f.write(f"- Y坐标范围: [{np.min(sensitive_states[:, 1]):.4f}, {np.max(sensitive_states[:, 1]):.4f}]\n")
            f.write(f"- Z坐标范围: [{np.min(sensitive_states[:, 2]):.4f}, {np.max(sensitive_states[:, 2]):.4f}]\n")
    else:
        print("未找到满足阈值的敏感区域点。")
    
    print("分析完成，结果已保存到 'stability_analysis.md'")
