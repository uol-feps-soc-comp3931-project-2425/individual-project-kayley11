import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lorenz_system import LorenzSystem
import pickle

def generate_reference_solution(t_span, initial_state, dt_output=0.01, rtol=1e-12, atol=1e-12):
    lorenz = LorenzSystem()
    t_eval = np.arange(t_span[0], t_span[1], dt_output)
    solution = solve_ivp(
        lorenz.derivatives,
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    return solution.t, solution.y.T

def save_reference_solution(filename, t, states):
    with open(filename, 'wb') as f:
        pickle.dump({'t': t, 'states': states}, f)

def load_reference_solution(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['t'], data['states']

def plot_reference_solution(t, states, title="Lorenz System Reference Solution (RK45)"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.savefig('reference_solution_3d.png', dpi=300)
    plt.close()
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t, states[:, 0], 'r-')
    axs[0].set_ylabel('X')
    axs[0].set_title('X component over time')
    axs[1].plot(t, states[:, 1], 'g-')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Y component over time')
    axs[2].plot(t, states[:, 2], 'b-')
    axs[2].set_ylabel('Z')
    axs[2].set_xlabel('Time')
    axs[2].set_title('Z component over time')
    plt.tight_layout()
    plt.savefig('reference_solution_time_series.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    t_span = [0, 100]
    initial_state = [1.0, 1.0, 1.0]
    dt_output = 0.01
    print("生成Lorenz系统的高精度参考解...")
    t, states = generate_reference_solution(t_span, initial_state, dt_output)
    save_reference_solution('reference_solution.pkl', t, states)
    print(f"参考解已保存到 'reference_solution.pkl'")
    print("绘制参考解...")
    plot_reference_solution(t, states)
    print("参考解图表已保存")
    print("\n参考解统计信息:")
    print(f"时间范围: {t[0]} 到 {t[-1]}")
    print(f"时间步数: {len(t)}")
    print(f"X范围: [{np.min(states[:, 0]):.4f}, {np.max(states[:, 0]):.4f}]")
    print(f"Y范围: [{np.min(states[:, 1]):.4f}, {np.max(states[:, 1]):.4f}]")
    print(f"Z范围: [{np.min(states[:, 2]):.4f}, {np.max(states[:, 2]):.4f}]")
    lorenz = LorenzSystem()
    pseudo_energies = np.array([lorenz.pseudo_energy(state) for state in states])
    print(f"伪能量范围: [{np.min(pseudo_energies):.4f}, {np.max(pseudo_energies):.4f}]")
    print(f"伪能量平均值: {np.mean(pseudo_energies):.4f}")
    print(f"伪能量标准差: {np.std(pseudo_energies):.4f}")
