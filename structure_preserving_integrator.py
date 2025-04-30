import numpy as np
import matplotlib.pyplot as plt
import os
import time
from lorenz_system import LorenzSystem

class StructurePreservingIntegrator:
    def __init__(self, system=None, enhanced=False):
        if system is None:
            self.system = LorenzSystem()
        else:
            self.system = system
        self.sigma = self.system.sigma
        self.rho = self.system.rho
        self.beta = self.system.beta
        self.theoretical_contraction_rate = -(self.sigma + 1 + self.beta)
        self.enhanced = enhanced
        self.stats = {
            'volume_corrections': 0,
            'constraint_enforcements': 0,
            'attractor_preservations': 0,
            'total_steps': 0
        }
    
    def step(self, t, state, dt):
        self.stats['total_steps'] += 1
        k1 = self.system.derivatives(t, state)
        k2 = self.system.derivatives(t + dt/2, state + dt*k1/2)
        k3 = self.system.derivatives(t + dt/2, state + dt*k2/2)
        k4 = self.system.derivatives(t + dt, state + dt*k3)
        next_state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        next_state = self.apply_structure_preserving_corrections(state, next_state, dt)
        return next_state
    
    def apply_structure_preserving_corrections(self, state, next_state, dt):
        next_state = self.apply_volume_contraction_correction(state, next_state, dt)
        next_state = self.apply_constraint_enforcement(next_state)
        next_state = self.apply_attractor_preservation(state, next_state, dt)
        return next_state
    
    def apply_volume_contraction_correction(self, state, next_state, dt):
        x, y, z = state
        jacobian_trace = -self.sigma - 1 - self.beta
        # Simplified numerical contraction calculation (may need refinement)
        state_norm = np.linalg.norm(state)
        if state_norm > 1e-10: # Avoid division by zero
             numerical_contraction = (np.linalg.norm(next_state) - state_norm) / (dt * state_norm)
        else:
             numerical_contraction = 0.0 # Assign a default value or handle appropriately

        correction_factor = np.exp(dt * (jacobian_trace - numerical_contraction))
        corrected_state = next_state * correction_factor
        self.stats['volume_corrections'] += 1
        return corrected_state
    
    def apply_constraint_enforcement(self, state):
        x, y, z = state
        if z < 0:
            z = 1e-6
            self.stats['constraint_enforcements'] += 1
        max_value = 1000
        if np.max(np.abs(state)) > max_value:
            scale_factor = max_value / np.max(np.abs(state))
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
            self.stats['constraint_enforcements'] += 1
        return np.array([x, y, z])
    
    def apply_attractor_preservation(self, state, next_state, dt):
        x, y, z = state
        next_x, next_y, next_z = next_state
        energy = x**2 + y**2 + (z - self.rho)**2
        next_energy = next_x**2 + next_y**2 + (next_z - self.rho)**2

        # Avoid division by zero or near-zero energy
        if energy > 1e-10:
            energy_change_rate = (next_energy - energy) / (dt * energy)
        else:
            energy_change_rate = 0.0 # Or handle appropriately

        if self.enhanced:
            if abs(energy_change_rate) > 0.1:
                if next_energy > 1e-10: # Avoid division by zero
                    correction_factor = np.sqrt(energy / next_energy)
                    next_state = next_state * correction_factor
                    self.stats['attractor_preservations'] += 1
        else:
            if energy_change_rate > 0.2:
                 if next_energy > 1e-10: # Avoid division by zero
                     correction_factor = np.sqrt(energy / next_energy)
                     next_state = next_state * correction_factor
                     self.stats['attractor_preservations'] += 1
        return next_state
    
    def solve(self, t_span, initial_state, dt):
        t_start, t_end = t_span
        t_current = t_start
        state_current = np.array(initial_state)
        t_points = [t_current]
        states = [state_current.copy()]
        self.stats = {
            'volume_corrections': 0,
            'constraint_enforcements': 0,
            'attractor_preservations': 0,
            'total_steps': 0
        }
        while t_current < t_end:
            state_current = self.step(t_current, state_current, dt)
            t_current += dt
            t_points.append(t_current)
            states.append(state_current.copy())
        return np.array(t_points), np.array(states)


class EnhancedStructurePreservingIntegrator(StructurePreservingIntegrator):
    def __init__(self, system=None):
        super().__init__(system, enhanced=True)

    def apply_volume_contraction_correction(self, state, next_state, dt):
        x, y, z = state
        jacobian = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        jacobian_trace = np.trace(jacobian)
        dx = next_state - state
        state_norm = np.linalg.norm(state)
        if state_norm > 1e-10: # Avoid division by zero
            numerical_contraction = np.sum(np.diag(jacobian)) * dt / state_norm
        else:
            numerical_contraction = 0.0 # Or handle appropriately

        correction_factor = np.exp(dt * (jacobian_trace - numerical_contraction))
        corrected_state = next_state * correction_factor
        self.stats['volume_corrections'] += 1
        return corrected_state
    
    def apply_constraint_enforcement(self, state):
        x, y, z = state
        if z < 0:
            energy = x**2 + y**2 + (z - self.rho)**2
            z = 1e-6
            remaining_energy = energy - (z - self.rho)**2
            if remaining_energy > 0:
                # Avoid division by zero
                xy_norm_sq = x**2 + y**2
                if xy_norm_sq > 1e-10:
                   scale_factor = np.sqrt(remaining_energy / xy_norm_sq)
                   x *= scale_factor
                   y *= scale_factor
                else:
                   # Handle case where x and y are both zero or very small
                   # Option: Set x and y based on some distribution or keep them zero
                   x, y = 0.0, 0.0 # Example: Keep them zero

            self.stats['constraint_enforcements'] += 1
        max_value = 1000
        max_abs_state = np.max(np.abs(state))
        if max_abs_state > max_value:
            scale_factor = max_value / max_abs_state
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
            self.stats['constraint_enforcements'] += 1
        return np.array([x, y, z])


def test_structure_preserving_integrator():
    print("测试结构保持积分器...")
    t_span = [0, 100]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    lorenz = LorenzSystem()
    rk4_solver = lambda system: RK4Solver(system)
    spi_solver = lambda system: StructurePreservingIntegrator(system)
    espi_solver = lambda system: EnhancedStructurePreservingIntegrator(system)
    print("  使用RK4求解器...")
    rk4 = rk4_solver(lorenz)
    t_rk4, states_rk4 = rk4.solve(t_span, initial_state, dt)
    print("  使用结构保持积分器...")
    spi = spi_solver(lorenz)
    t_spi, states_spi = spi.solve(t_span, initial_state, dt)
    print("  使用增强版结构保持积分器...")
    espi = espi_solver(lorenz)
    t_espi, states_espi = espi.solve(t_span, initial_state, dt)
    theoretical_contraction = -(lorenz.sigma + 1 + lorenz.beta)
    volume_changes_rk4 = []
    for i in range(1, len(states_rk4)-1):
        v1 = states_rk4[i] - states_rk4[i-1]
        v2 = states_rk4[i+1] - states_rk4[i]
        v3 = np.cross(v1, v2)
        # Ensure states_rk4[i] is not zero vector before dot product
        if np.linalg.norm(states_rk4[i]) > 1e-10:
             volume = np.abs(np.dot(v3, states_rk4[i])) / 6
             volume_changes_rk4.append(volume)
        else:
             volume_changes_rk4.append(0.0) # Handle zero state vector case

    volume_ratios_rk4 = np.diff(volume_changes_rk4) / (np.array(volume_changes_rk4[:-1]) + 1e-10)
    avg_volume_contraction_rk4 = np.mean(volume_ratios_rk4)
    volume_changes_spi = []
    for i in range(1, len(states_spi)-1):
        v1 = states_spi[i] - states_spi[i-1]
        v2 = states_spi[i+1] - states_spi[i]
        v3 = np.cross(v1, v2)
        if np.linalg.norm(states_spi[i]) > 1e-10:
             volume = np.abs(np.dot(v3, states_spi[i])) / 6
             volume_changes_spi.append(volume)
        else:
             volume_changes_spi.append(0.0)

    volume_ratios_spi = np.diff(volume_changes_spi) / (np.array(volume_changes_spi[:-1]) + 1e-10)
    avg_volume_contraction_spi = np.mean(volume_ratios_spi)
    volume_changes_espi = []
    for i in range(1, len(states_espi)-1):
        v1 = states_espi[i] - states_espi[i-1]
        v2 = states_espi[i+1] - states_espi[i]
        v3 = np.cross(v1, v2)
        if np.linalg.norm(states_espi[i]) > 1e-10:
            volume = np.abs(np.dot(v3, states_espi[i])) / 6
            volume_changes_espi.append(volume)
        else:
            volume_changes_espi.append(0.0)

    volume_ratios_espi = np.diff(volume_changes_espi) / (np.array(volume_changes_espi[:-1]) + 1e-10)
    avg_volume_contraction_espi = np.mean(volume_ratios_espi)
    energy_rk4 = states_rk4[:, 0]**2 + states_rk4[:, 1]**2 + (states_rk4[:, 2] - lorenz.rho)**2
    energy_spi = states_spi[:, 0]**2 + states_spi[:, 1]**2 + (states_spi[:, 2] - lorenz.rho)**2
    energy_espi = states_espi[:, 0]**2 + states_espi[:, 1]**2 + (states_espi[:, 2] - lorenz.rho)**2
    energy_variation_rk4 = np.std(energy_rk4) / np.mean(energy_rk4)
    energy_variation_spi = np.std(energy_spi) / np.mean(energy_spi)
    energy_variation_espi = np.std(energy_espi) / np.mean(energy_espi)
    spi_stats = {
        'volume': spi.stats['volume_corrections'] / spi.stats['total_steps'],
        'constraint': spi.stats['constraint_enforcements'] / spi.stats['total_steps'],
        'attractor': spi.stats['attractor_preservations'] / spi.stats['total_steps']
    }
    espi_stats = {
        'volume': espi.stats['volume_corrections'] / espi.stats['total_steps'],
        'constraint': espi.stats['constraint_enforcements'] / espi.stats['total_steps'],
        'attractor': espi.stats['attractor_preservations'] / espi.stats['total_steps']
    }
    with open('structure_preserving_performance.md', 'w') as f:
        f.write("# 结构保持积分器性能对比\n\n")
        f.write("## 评估参数\n")
        f.write(f"- 时间范围: {t_span}\n")
        f.write(f"- 初始状态: {initial_state}\n")
        f.write(f"- 时间步长: {dt}\n\n")
        f.write("## 体积收缩率对比\n\n")
        f.write("| Solver | Volume Contraction Rate | Error from Theoretical |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Theoretical | {theoretical_contraction:.6f} | {0.0:.6f} |\n")
        f.write(f"| RK4 | {avg_volume_contraction_rk4:.6f} | {abs(avg_volume_contraction_rk4 - theoretical_contraction):.6f} |\n")
        f.write(f"| Structure Preserving | {avg_volume_contraction_spi:.6f} | {abs(avg_volume_contraction_spi - theoretical_contraction):.6f} |\n")
        f.write(f"| Enhanced Structure Preserving | {avg_volume_contraction_espi:.6f} | {abs(avg_volume_contraction_espi - theoretical_contraction):.6f} |\n\n")
        f.write("## 伪能量稳定性对比\n\n")
        f.write("| Solver | Energy Variation Coefficient |\n")
        f.write("| --- | --- |\n")
        f.write(f"| RK4 | {energy_variation_rk4:.6f} |\n")
        f.write(f"| Structure Preserving | {energy_variation_spi:.6f} |\n")
        f.write(f"| Enhanced Structure Preserving | {energy_variation_espi:.6f} |\n\n")
        f.write("## 结构保持积分器修正统计\n\n")
        f.write("| Correction Type | Structure Preserving | Enhanced Structure Preserving |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Volume Correction | {spi_stats['volume']*100:.2f}% | {espi_stats['volume']*100:.2f}% |\n")
        f.write(f"| Constraint Enforcement | {spi_stats['constraint']*100:.2f}% | {espi_stats['constraint']*100:.2f}% |\n")
        f.write(f"| Attractor Preservation | {spi_stats['attractor']*100:.2f}% | {espi_stats['attractor']*100:.2f}% |\n")
    print("测试完成，结果已保存到 'structure_preserving_performance.md'")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_rk4[:, 0], states_rk4[:, 1], states_rk4[:, 2], 'b-', linewidth=0.5, label='RK4')
    ax.plot(states_spi[:, 0], states_spi[:, 1], states_spi[:, 2], 'r-', linewidth=0.5, label='Structure Preserving')
    ax.plot(states_espi[:, 0], states_espi[:, 1], states_espi[:, 2], 'g-', linewidth=0.5, label='Enhanced Structure Preserving')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    plt.savefig('structure_preserving_trajectory.png', dpi=300)
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.plot(t_rk4, energy_rk4, 'b-', label='RK4')
    plt.plot(t_spi, energy_spi, 'r-', label='Structure Preserving')
    plt.plot(t_espi, energy_espi, 'g-', label='Enhanced Structure Preserving')
    plt.xlabel('Time')
    plt.ylabel('Pseudo-Energy')
    plt.title('Pseudo-Energy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('structure_preserving_energy.png', dpi=300)
    plt.close()

class RK4Solver:
    def __init__(self, system):
        self.system = system
    
    def step(self, t, state, dt):
        k1 = self.system.derivatives(t, state)
        k2 = self.system.derivatives(t + dt/2, state + dt*k1/2)
        k3 = self.system.derivatives(t + dt/2, state + dt*k2/2)
        k4 = self.system.derivatives(t + dt, state + dt*k3)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def solve(self, t_span, initial_state, dt):
        t_start, t_end = t_span
        t_current = t_start
        state_current = np.array(initial_state)
        t_points = [t_current]
        states = [state_current.copy()]
        while t_current < t_end:
            state_current = self.step(t_current, state_current, dt)
            t_current += dt
            t_points.append(t_current)
            states.append(state_current.copy())
        return np.array(t_points), np.array(states)

if __name__ == "__main__":
    test_structure_preserving_integrator()
