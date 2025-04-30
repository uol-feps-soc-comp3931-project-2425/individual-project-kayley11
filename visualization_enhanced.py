import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import pickle
from lorenz_system import LorenzSystem
from numerical_solvers import (
    EulerSolver, MidpointSolver, RK2Solver, RK4Solver,
    SymplecticEulerSolver, EnergyPreservingSolver
)
from hybrid_solver import HybridAdaptiveSolver
from structure_preserving_integrator import StructurePreservingIntegrator

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class LorenzVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.cmap = cm.viridis
        self.solver_names = {
            "Euler": "欧拉法",
            "Midpoint": "中点法",
            "RK2": "RK2方法",
            "RK4": "RK4方法",
            "Symplectic": "辛方法",
            "Energy": "能量守恒方法",
            "Hybrid": "混合自适应求解器",
            "Structure": "结构保持积分器"
        }

    def plot_3d_trajectory(self, t, states, title="Lorenz System Trajectory",
                          filename="trajectory_3d.png", show_time_color=True,
                          azimuth=-60, elevation=30):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if show_time_color:
            norm = Normalize(t.min(), t.max())
            colors = self.cmap(norm(t))
            for i in range(len(t)-1):
                ax.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                       color=colors[i], linewidth=1.0)
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Time')
        else:
            ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=1.0)

        ax.scatter(states[0, 0], states[0, 1], states[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], c='r', marker='o', s=100, label='End')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        ax.view_init(elevation, azimuth)

        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def create_trajectory_animation(self, t, states, title="Lorenz System Trajectory Animation",
                                   filename="trajectory_animation.gif", fps=30, dpi=100):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim([np.min(states[:, 0]), np.max(states[:, 0])])
        ax.set_ylim([np.min(states[:, 1]), np.max(states[:, 1])])
        ax.set_zlim([np.min(states[:, 2]), np.max(states[:, 2])])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        line, = ax.plot([], [], [], 'b-', linewidth=1.0)
        point, = ax.plot([], [], [], 'ro', markersize=6)
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            time_text.set_text('')
            return line, point, time_text

        def update(frame):
            i = frame
            line.set_data(states[:i, 0], states[:i, 1])
            line.set_3d_properties(states[:i, 2])
            point.set_data([states[i-1, 0]], [states[i-1, 1]])
            point.set_3d_properties([states[i-1, 2]])
            time_text.set_text(f'Time: {t[i-1]:.2f}')
            return line, point, time_text

        frames = min(len(t), 500)
        step = max(1, len(t) // frames)
        ani = FuncAnimation(fig, update, frames=range(1, len(t), step),
                           init_func=init, blit=True, interval=1000/fps)

        ani.save(os.path.join(self.output_dir, filename), writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"动画已保存到 {os.path.join(self.output_dir, filename)}")

    def plot_phase_portraits(self, t, states, title="Lorenz System Phase Portraits",
                            filename="phase_portraits.png"):
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        norm = Normalize(t.min(), t.max())
        colors = self.cmap(norm(t))

        for i in range(len(t)-1):
            axs[0, 0].plot(states[i:i+2, 0], states[i:i+2, 1], color=colors[i], linewidth=0.8)
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        axs[0, 0].set_title('X-Y Plane')

        for i in range(len(t)-1):
            axs[0, 1].plot(states[i:i+2, 0], states[i:i+2, 2], color=colors[i], linewidth=0.8)
        axs[0, 1].set_xlabel('X')
        axs[0, 1].set_ylabel('Z')
        axs[0, 1].set_title('X-Z Plane')

        for i in range(len(t)-1):
            axs[1, 0].plot(states[i:i+2, 1], states[i:i+2, 2], color=colors[i], linewidth=0.8)
        axs[1, 0].set_xlabel('Y')
        axs[1, 0].set_ylabel('Z')
        axs[1, 0].set_title('Y-Z Plane')

        for i in range(3):
            axs[1, 1].plot(t, states[:, i], label=f'{"XYZ"[i]}-coordinate')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].set_title('Time Series')
        axs[1, 1].legend()

        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axs.ravel().tolist(), pad=0.01)
        cbar.set_label('Time')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_attractor_structure(self, t, states, title="Lorenz Attractor Structure",
                               filename="attractor_structure.png"):
        fig = plt.figure(figsize=(15, 12))

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D View')

        ax2 = fig.add_subplot(222)
        h = ax2.hist2d(states[:, 0], states[:, 1], bins=50, cmap='viridis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Density in X-Y Plane')
        plt.colorbar(h[3], ax=ax2)

        ax3 = fig.add_subplot(223)
        h = ax3.hist2d(states[:, 0], states[:, 2], bins=50, cmap='viridis')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('Density in X-Z Plane')
        plt.colorbar(h[3], ax=ax3)

        ax4 = fig.add_subplot(224)
        h = ax4.hist2d(states[:, 1], states[:, 2], bins=50, cmap='viridis')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('Density in Y-Z Plane')
        plt.colorbar(h[3], ax=ax4)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def compare_trajectories(self, results_dict, title="Comparison of Different Solvers",
                            filename="trajectory_comparison.png"):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange']

        for i, (name, result) in enumerate(results_dict.items()):
            t = result['t']
            states = result['states']
            display_name = self.solver_names.get(name, name)
            ax.plot(states[:, 0], states[:, 1], states[:, 2],
                    color=colors[i % len(colors)], linewidth=1.0, label=display_name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=16, fontfamily='DejaVu Sans')
        ax.legend(loc='upper right')

        print(f"正在保存图像: {filename}, 标题设置为: {title}")

        full_path = os.path.join(self.output_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图像已保存到: {full_path}")

    def create_multi_view_animation(self, t, states, title="Multi-View Animation",
                                  filename="multi_view_animation.gif", fps=20, dpi=100):
        fig = plt.figure(figsize=(15, 12))

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_xlim([np.min(states[:, 0]), np.max(states[:, 0])])
        ax1.set_ylim([np.min(states[:, 1]), np.max(states[:, 1])])
        ax1.set_zlim([np.min(states[:, 2]), np.max(states[:, 2])])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory')

        ax2 = fig.add_subplot(222)
        ax2.set_xlim([np.min(states[:, 0]), np.max(states[:, 0])])
        ax2.set_ylim([np.min(states[:, 1]), np.max(states[:, 1])])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('X-Y Plane')

        ax3 = fig.add_subplot(223)
        ax3.set_xlim([np.min(states[:, 0]), np.max(states[:, 0])])
        ax3.set_ylim([np.min(states[:, 2]), np.max(states[:, 2])])
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('X-Z Plane')

        ax4 = fig.add_subplot(224)
        ax4.set_xlim([np.min(states[:, 1]), np.max(states[:, 1])])
        ax4.set_ylim([np.min(states[:, 2]), np.max(states[:, 2])])
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('Y-Z Plane')

        line1, = ax1.plot([], [], [], 'b-', linewidth=1.0)
        point1, = ax1.plot([], [], [], 'ro', markersize=6)
        line2, = ax2.plot([], [], 'b-', linewidth=1.0)
        point2, = ax2.plot([], [], 'ro', markersize=6)
        line3, = ax3.plot([], [], 'b-', linewidth=1.0)
        point3, = ax3.plot([], [], 'ro', markersize=6)
        line4, = ax4.plot([], [], 'b-', linewidth=1.0)
        point4, = ax4.plot([], [], 'ro', markersize=6)

        time_text = fig.text(0.5, 0.01, '', ha='center')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        def init():
            line1.set_data([], [])
            line1.set_3d_properties([])
            point1.set_data([], [])
            point1.set_3d_properties([])
            line2.set_data([], [])
            point2.set_data([], [])
            line3.set_data([], [])
            point3.set_data([], [])
            line4.set_data([], [])
            point4.set_data([], [])
            time_text.set_text('')
            return line1, point1, line2, point2, line3, point3, line4, point4, time_text

        def update(frame):
            i = frame
            line1.set_data(states[:i, 0], states[:i, 1])
            line1.set_3d_properties(states[:i, 2])
            point1.set_data([states[i-1, 0]], [states[i-1, 1]])
            point1.set_3d_properties([states[i-1, 2]])
            line2.set_data(states[:i, 0], states[:i, 1])
            point2.set_data([states[i-1, 0]], [states[i-1, 1]])
            line3.set_data(states[:i, 0], states[:i, 2])
            point3.set_data([states[i-1, 0]], [states[i-1, 2]])
            line4.set_data(states[:i, 1], states[:i, 2])
            point4.set_data([states[i-1, 1]], [states[i-1, 2]])
            time_text.set_text(f'Time: {t[i-1]:.2f}')
            return line1, point1, line2, point2, line3, point3, line4, point4, time_text

        frames = min(len(t), 500)
        step = max(1, len(t) // frames)
        ani = FuncAnimation(fig, update, frames=range(1, len(t), step),
                           init_func=init, blit=True, interval=1000/fps)

        ani.save(os.path.join(self.output_dir, filename), writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"多视角动画已保存到 {os.path.join(self.output_dir, filename)}")

    def create_rotating_view_animation(self, states, title="Rotating View Animation",
                                     filename="rotating_view_animation.gif", fps=20, dpi=100):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        def init():
            return []

        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return []

        ani = FuncAnimation(fig, update, frames=range(0, 360, 2),
                           init_func=init, blit=True, interval=1000/fps)

        ani.save(os.path.join(self.output_dir, filename), writer='pillow', fps=fps, dpi=dpi)
        plt.close()
        print(f"旋转视角动画已保存到 {os.path.join(self.output_dir, filename)}")


def generate_visualization_data():
    print("生成可视化数据...")
    t_span = [0, 50]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01

    lorenz = LorenzSystem()
    solvers = {
        "Euler": EulerSolver(lorenz),
        "Midpoint": MidpointSolver(lorenz),
        "RK2": RK2Solver(lorenz),
        "RK4": RK4Solver(lorenz),
        "Symplectic": SymplecticEulerSolver(lorenz),
        "Energy": EnergyPreservingSolver(lorenz),
        "Hybrid": HybridAdaptiveSolver(lorenz),
        "Structure": StructurePreservingIntegrator(lorenz)
    }

    results = {}
    for name, solver in solvers.items():
        print(f"  使用 {name} 求解器...")
        t, states = solver.solve(t_span, initial_state, dt)
        results[name] = {'t': t, 'states': states}

    with open('visualization_data.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"可视化数据已保存到 'visualization_data.pkl'")
    return results


def load_visualization_data():
    try:
        with open('visualization_data.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"已加载可视化数据，包含 {len(results)} 个求解器的结果")
        return results
    except FileNotFoundError:
        print("未找到保存的可视化数据，正在生成...")
        return generate_visualization_data()


def create_basic_visualizations(results=None):
    print("创建基本可视化...")
    if results is None:
        results = load_visualization_data()

    visualizer = LorenzVisualizer()
    rk4_result = results["RK4"]
    t_rk4 = rk4_result['t']
    states_rk4 = rk4_result['states']

    visualizer.plot_3d_trajectory(t_rk4, states_rk4,
                                 title="Lorenz System 3D Trajectory (RK4)",
                                 filename="rk4_trajectory_3d.png")
    visualizer.plot_phase_portraits(t_rk4, states_rk4,
                                  title="Lorenz System Phase Portraits (RK4)",
                                  filename="rk4_phase_portraits.png")
    visualizer.plot_attractor_structure(t_rk4, states_rk4,
                                      title="Lorenz Attractor Structure (RK4)",
                                      filename="rk4_attractor_structure.png")
    visualizer.compare_trajectories(results,
                                  title="Comparison of Different Solvers",
                                  filename="solvers_comparison.png")

    print("基本可视化已完成")


def create_animations(results=None):
    print("创建动画可视化...")
    if results is None:
        results = load_visualization_data()

    visualizer = LorenzVisualizer()
    rk4_result = results["RK4"]
    t_rk4 = rk4_result['t']
    states_rk4 = rk4_result['states']

    visualizer.create_trajectory_animation(t_rk4, states_rk4,
                                         title="Lorenz System Trajectory Animation (RK4)",
                                         filename="rk4_trajectory_animation.gif")
    visualizer.create_multi_view_animation(t_rk4, states_rk4,
                                         title="Lorenz System Multi-View Animation (RK4)",
                                         filename="rk4_multi_view_animation.gif")
    visualizer.create_rotating_view_animation(states_rk4,
                                            title="Lorenz System Rotating View (RK4)",
                                            filename="rk4_rotating_view_animation.gif")

    print("动画可视化已完成")


if __name__ == "__main__":
    results = generate_visualization_data()
    create_basic_visualizations(results)
    create_animations(results)
