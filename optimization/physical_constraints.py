import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy.optimize import minimize
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("physical_constraints.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhysicalConstraints")

class ConstraintType(Enum):
    ENERGY = 1
    VOLUME_CONTRACTION = 2
    DISSIPATION = 3
    ATTRACTOR_TOPOLOGY = 4
    LYAPUNOV_STABILITY = 5
    PHASE_SPACE_BOUNDS = 6
    STATISTICAL_INVARIANTS = 7

class ConstraintEnforcementMethod(Enum):
    PROJECTION = 1
    LAGRANGE_MULTIPLIER = 2
    PENALTY = 3
    SOFT_CONSTRAINT = 4
    RELAXATION = 5
    OPTIMIZATION = 6

class PhysicalConstraint:
    def __init__(self, name, constraint_type, enforcement_method=ConstraintEnforcementMethod.PROJECTION):
        self.name = name
        self.constraint_type = constraint_type
        self.enforcement_method = enforcement_method
        self.stats = {
            'enforcements': 0,
            'violations': 0,
            'total_checks': 0,
            'max_violation': 0.0,
            'avg_violation': 0.0
        }
    def check_constraint(self, t, state, system):
        self.stats['total_checks'] += 1
        return True, 0.0
    def enforce_constraint(self, t, state, next_state, dt, system):
        return next_state
    def get_stats_report(self):
        if self.stats['total_checks'] == 0:
            return f"{self.name} 约束尚未检查"
        violation_rate = self.stats['violations'] / self.stats['total_checks'] * 100
        enforcement_rate = self.stats['enforcements'] / max(1, self.stats['violations']) * 100
        report = f"{self.name} 约束统计报告:\n"
        report += f"总检查次数: {self.stats['total_checks']}\n"
        report += f"违反次数: {self.stats['violations']} ({violation_rate:.2f}%)\n"
        report += f"实施次数: {self.stats['enforcements']} ({enforcement_rate:.2f}%)\n"
        report += f"最大违反程度: {self.stats['max_violation']:.6e}\n"
        report += f"平均违反程度: {self.stats['avg_violation']:.6e}\n"
        return report
    def reset_stats(self):
        self.stats = {
            'enforcements': 0,
            'violations': 0,
            'total_checks': 0,
            'max_violation': 0.0,
            'avg_violation': 0.0
        }

class EnergyConstraint(PhysicalConstraint):
    def __init__(self, enforcement_method=ConstraintEnforcementMethod.PROJECTION,
                 tolerance=1e-3, strict=False):
        super().__init__("能量", ConstraintType.ENERGY, enforcement_method)
        self.tolerance = tolerance
        self.strict = strict
        self.energy_history = []
        self.time_history = []
    def calculate_energy(self, state, system):
        if hasattr(system, 'pseudo_energy'):
            return system.pseudo_energy(state)
        else:
            x, y, z = state
            rho = getattr(system, 'rho', 28.0)
            return 0.5 * (x**2 + y**2 + (z - rho)**2)
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        current_energy = self.calculate_energy(state, system)
        self.energy_history.append(current_energy)
        self.time_history.append(t)
        if len(self.energy_history) < 2:
            return True, 0.0
        prev_energy = self.energy_history[-2]
        energy_change = current_energy - prev_energy
        if self.strict:
            is_satisfied = abs(energy_change) <= self.tolerance
            violation = abs(energy_change)
        else:
            is_satisfied = energy_change <= self.tolerance
            violation = max(0, energy_change)
        if not is_satisfied:
            self.stats['violations'] += 1
            self.stats['max_violation'] = max(self.stats['max_violation'], violation)
            n_violations = self.stats['violations']
            self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + violation) / n_violations
        return is_satisfied, violation
    def enforce_constraint(self, t, state, next_state, dt, system):
        current_energy = self.calculate_energy(state, system)
        next_energy = self.calculate_energy(next_state, system)
        energy_change = next_energy - current_energy
        if (self.strict and abs(energy_change) > self.tolerance) or \
           (not self.strict and energy_change > self.tolerance):
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                if next_energy > 0:
                    scale_factor = np.sqrt(current_energy / next_energy)
                    corrected_state = next_state * scale_factor
                else:
                    corrected_state = next_state.copy()
            elif self.enforcement_method == ConstraintEnforcementMethod.LAGRANGE_MULTIPLIER:
                x, y, z = next_state
                rho = getattr(system, 'rho', 28.0)
                gradient = np.array([x, y, z - rho])
                gradient_norm_sq = np.sum(gradient**2)
                if gradient_norm_sq > 1e-10:
                    lambda_factor = energy_change / gradient_norm_sq
                    corrected_state = next_state - lambda_factor * gradient
                else:
                    corrected_state = next_state.copy()
            elif self.enforcement_method == ConstraintEnforcementMethod.OPTIMIZATION:
                def objective(s):
                    return np.sum((s - next_state)**2)
                def constraint(s):
                    e = self.calculate_energy(s, system)
                    if self.strict:
                        return current_energy - e
                    else:
                        return e - current_energy
                result = minimize(
                    objective,
                    next_state,
                    constraints={'type': 'eq' if self.strict else 'ineq', 'fun': constraint},
                    method='SLSQP'
                )
                if result.success:
                    corrected_state = result.x
                else:
                    if next_energy > 0:
                        scale_factor = np.sqrt(current_energy / next_energy)
                        corrected_state = next_state * scale_factor
                    else:
                        corrected_state = next_state.copy()
            else:
                if next_energy > 0:
                    ideal_factor = np.sqrt(current_energy / next_energy)
                    soft_factor = 0.5
                    effective_factor = 1.0 + soft_factor * (ideal_factor - 1.0)
                    corrected_state = next_state * effective_factor
                else:
                    corrected_state = next_state.copy()
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_energy_history(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.energy_history:
            logger.warning("能量历史为空，无法绘图")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.energy_history)
        plt.xlabel('时间')
        plt.ylabel('能量')
        plt.title('系统能量历史')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "energy_history.png"), dpi=300)
        plt.close()
        if len(self.energy_history) > 1:
            energy_changes = np.diff(self.energy_history)
            time_diffs = np.diff(self.time_history)
            energy_change_rates = energy_changes / time_diffs
            plt.figure(figsize=(12, 6))
            plt.plot(self.time_history[1:], energy_change_rates)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('时间')
            plt.ylabel('能量变化率')
            plt.title('系统能量变化率')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "energy_change_rate.png"), dpi=300)
            plt.close()

class VolumeContractionConstraint(PhysicalConstraint):
    def __init__(self, enforcement_method=ConstraintEnforcementMethod.PROJECTION,
                 tolerance=1e-3):
        super().__init__("体积收缩率", ConstraintType.VOLUME_CONTRACTION, enforcement_method)
        self.tolerance = tolerance
        self.contraction_history = []
        self.time_history = []
    def calculate_theoretical_contraction(self, system):
        if hasattr(system, 'volume_contraction'):
            return system.volume_contraction()
        else:
            sigma = getattr(system, 'sigma', 10.0)
            beta = getattr(system, 'beta', 8/3)
            return -(sigma + 1 + beta)
    def calculate_numerical_contraction(self, state, next_state, dt):
        state_norm = np.linalg.norm(state)
        next_norm = np.linalg.norm(next_state)
        if state_norm > 1e-10:
            return (next_norm - state_norm) / (dt * state_norm)
        else:
            return 0.0
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        if not hasattr(self, 'prev_state') or self.prev_state is None:
            self.prev_state = state.copy()
            self.prev_time = t
            return True, 0.0
        dt = t - self.prev_time
        if dt > 1e-10:
            numerical_contraction = self.calculate_numerical_contraction(
                self.prev_state, state, dt
            )
            theoretical_contraction = self.calculate_theoretical_contraction(system)
            self.contraction_history.append(numerical_contraction)
            self.time_history.append(t)
            contraction_error = abs(numerical_contraction - theoretical_contraction)
            is_satisfied = contraction_error <= self.tolerance
            if not is_satisfied:
                self.stats['violations'] += 1
                self.stats['max_violation'] = max(self.stats['max_violation'], contraction_error)
                n_violations = self.stats['violations']
                self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + contraction_error) / n_violations
            self.prev_state = state.copy()
            self.prev_time = t
            return is_satisfied, contraction_error
        self.prev_state = state.copy()
        self.prev_time = t
        return True, 0.0
    def enforce_constraint(self, t, state, next_state, dt, system):
        numerical_contraction = self.calculate_numerical_contraction(state, next_state, dt)
        theoretical_contraction = self.calculate_theoretical_contraction(system)
        contraction_error = abs(numerical_contraction - theoretical_contraction)
        if contraction_error > self.tolerance:
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                correction_factor = np.exp(dt * (theoretical_contraction - numerical_contraction))
                corrected_state = next_state * correction_factor
            elif self.enforcement_method == ConstraintEnforcementMethod.LAGRANGE_MULTIPLIER:
                gradient = next_state / np.linalg.norm(next_state) if np.linalg.norm(next_state) > 1e-10 else np.ones_like(next_state)
                lambda_factor = dt * (theoretical_contraction - numerical_contraction)
                corrected_state = next_state * (1 + lambda_factor)
            elif self.enforcement_method == ConstraintEnforcementMethod.OPTIMIZATION:
                def objective(s):
                    return np.sum((s - next_state)**2)
                def constraint(s):
                    num_contr = self.calculate_numerical_contraction(state, s, dt)
                    return self.tolerance - abs(num_contr - theoretical_contraction)
                result = minimize(
                    objective,
                    next_state,
                    constraints={'type': 'ineq', 'fun': constraint},
                    method='SLSQP'
                )
                if result.success:
                    corrected_state = result.x
                else:
                    correction_factor = np.exp(dt * (theoretical_contraction - numerical_contraction))
                    corrected_state = next_state * correction_factor
            else:
                ideal_factor = np.exp(dt * (theoretical_contraction - numerical_contraction))
                soft_factor = 0.5
                effective_factor = 1.0 + soft_factor * (ideal_factor - 1.0)
                corrected_state = next_state * effective_factor
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_contraction_history(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.contraction_history:
            logger.warning("体积收缩率历史为空，无法绘图")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.contraction_history)
        if hasattr(self, 'system'):
            theoretical_contraction = self.calculate_theoretical_contraction(self.system)
            plt.axhline(y=theoretical_contraction, color='r', linestyle='--', label='理论值')
        plt.xlabel('时间')
        plt.ylabel('体积收缩率')
        plt.title('系统体积收缩率历史')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "volume_contraction_history.png"), dpi=300)
        plt.close()

class DissipationConstraint(PhysicalConstraint):
    def __init__(self, enforcement_method=ConstraintEnforcementMethod.SOFT_CONSTRAINT,
                 tolerance=1e-3):
        super().__init__("耗散率", ConstraintType.DISSIPATION, enforcement_method)
        self.tolerance = tolerance
        self.dissipation_history = []
        self.time_history = []
    def calculate_theoretical_dissipation(self, state, system):
        if hasattr(system, 'dissipation_rate'):
            return system.dissipation_rate(state)
        else:
            x, y, z = state
            sigma = getattr(system, 'sigma', 10.0)
            beta = getattr(system, 'beta', 8/3)
            return -sigma * x**2 - y**2 - beta * z**2
    def calculate_numerical_dissipation(self, state, next_state, dt, system):
        if hasattr(system, 'pseudo_energy'):
            energy_current = system.pseudo_energy(state)
            energy_next = system.pseudo_energy(next_state)
        else:
            x, y, z = state
            nx, ny, nz = next_state
            rho = getattr(system, 'rho', 28.0)
            energy_current = 0.5 * (x**2 + y**2 + (z - rho)**2)
            energy_next = 0.5 * (nx**2 + ny**2 + (nz - rho)**2)
        return (energy_next - energy_current) / dt
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        if not hasattr(self, 'prev_state') or self.prev_state is None:
            self.prev_state = state.copy()
            self.prev_time = t
            return True, 0.0
        dt = t - self.prev_time
        if dt > 1e-10:
            numerical_dissipation = self.calculate_numerical_dissipation(
                self.prev_state, state, dt, system
            )
            theoretical_dissipation = self.calculate_theoretical_dissipation(self.prev_state, system)
            self.dissipation_history.append(numerical_dissipation)
            self.time_history.append(t)
            dissipation_error = numerical_dissipation - theoretical_dissipation
            is_satisfied = dissipation_error <= self.tolerance
            if not is_satisfied:
                self.stats['violations'] += 1
                self.stats['max_violation'] = max(self.stats['max_violation'], dissipation_error)
                n_violations = self.stats['violations']
                self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + dissipation_error) / n_violations
            self.prev_state = state.copy()
            self.prev_time = t
            return is_satisfied, dissipation_error
        self.prev_state = state.copy()
        self.prev_time = t
        return True, 0.0
    def enforce_constraint(self, t, state, next_state, dt, system):
        numerical_dissipation = self.calculate_numerical_dissipation(state, next_state, dt, system)
        theoretical_dissipation = self.calculate_theoretical_dissipation(state, system)
        dissipation_error = numerical_dissipation - theoretical_dissipation
        if dissipation_error > self.tolerance:
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                if hasattr(system, 'pseudo_energy'):
                    energy_current = system.pseudo_energy(state)
                    energy_next = system.pseudo_energy(next_state)
                else:
                    x, y, z = state
                    nx, ny, nz = next_state
                    rho = getattr(system, 'rho', 28.0)
                    energy_current = 0.5 * (x**2 + y**2 + (z - rho)**2)
                    energy_next = 0.5 * (nx**2 + ny**2 + (nz - rho)**2)
                target_energy = energy_current + theoretical_dissipation * dt
                if energy_next > 0 and target_energy > 0:
                    scale_factor = np.sqrt(target_energy / energy_next)
                    corrected_state = next_state * scale_factor
                else:
                    corrected_state = next_state.copy()
            elif self.enforcement_method == ConstraintEnforcementMethod.SOFT_CONSTRAINT:
                if hasattr(system, 'pseudo_energy'):
                    energy_current = system.pseudo_energy(state)
                    energy_next = system.pseudo_energy(next_state)
                else:
                    x, y, z = state
                    nx, ny, nz = next_state
                    rho = getattr(system, 'rho', 28.0)
                    energy_current = 0.5 * (x**2 + y**2 + (z - rho)**2)
                    energy_next = 0.5 * (nx**2 + ny**2 + (nz - rho)**2)
                target_energy = energy_current + theoretical_dissipation * dt
                if energy_next > 0 and target_energy > 0:
                    ideal_factor = np.sqrt(target_energy / energy_next)
                    soft_factor = 0.5
                    effective_factor = 1.0 + soft_factor * (ideal_factor - 1.0)
                    corrected_state = next_state * effective_factor
                else:
                    corrected_state = next_state.copy()
            else:
                if hasattr(system, 'pseudo_energy'):
                    energy_current = system.pseudo_energy(state)
                    energy_next = system.pseudo_energy(next_state)
                else:
                    x, y, z = state
                    nx, ny, nz = next_state
                    rho = getattr(system, 'rho', 28.0)
                    energy_current = 0.5 * (x**2 + y**2 + (z - rho)**2)
                    energy_next = 0.5 * (nx**2 + ny**2 + (nz - rho)**2)
                target_energy = energy_current + theoretical_dissipation * dt
                if energy_next > 0 and target_energy > 0:
                    ideal_factor = np.sqrt(target_energy / energy_next)
                    soft_factor = 0.3
                    effective_factor = 1.0 + soft_factor * (ideal_factor - 1.0)
                    corrected_state = next_state * effective_factor
                else:
                    corrected_state = next_state.copy()
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_dissipation_history(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.dissipation_history:
            logger.warning("耗散率历史为空，无法绘图")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.dissipation_history)
        plt.xlabel('时间')
        plt.ylabel('耗散率')
        plt.title('系统耗散率历史')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "dissipation_history.png"), dpi=300)
        plt.close()

class AttractorTopologyConstraint(PhysicalConstraint):
    def __init__(self, enforcement_method=ConstraintEnforcementMethod.SOFT_CONSTRAINT,
                 tolerance=1e-2):
        super().__init__("吸引子拓扑", ConstraintType.ATTRACTOR_TOPOLOGY, enforcement_method)
        self.tolerance = tolerance
        self.trajectory_history = []
        self.time_history = []
        self.attractor_features = {
            'center_of_mass': None,
            'radius_of_gyration': None,
            'principal_axes': None,
            'winding_number': None
        }
    def update_attractor_features(self, trajectory):
        if len(trajectory) < 10:
            return
        center_of_mass = np.mean(trajectory, axis=0)
        centered_trajectory = trajectory - center_of_mass
        radius_of_gyration = np.sqrt(np.mean(np.sum(centered_trajectory**2, axis=1)))
        covariance = np.cov(centered_trajectory, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        xy_trajectory = centered_trajectory[:, :2]
        angles = np.arctan2(xy_trajectory[:, 1], xy_trajectory[:, 0])
        angle_diff = np.diff(np.unwrap(angles))
        winding_number = np.sum(angle_diff) / (2 * np.pi)
        self.attractor_features['center_of_mass'] = center_of_mass
        self.attractor_features['radius_of_gyration'] = radius_of_gyration
        self.attractor_features['principal_axes'] = eigenvectors
        self.attractor_features['winding_number'] = winding_number
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        self.trajectory_history.append(state.copy())
        self.time_history.append(t)
        if len(self.trajectory_history) < 100:
            return True, 0.0
        if len(self.trajectory_history) % 100 == 0:
            recent_trajectory = np.array(self.trajectory_history[-min(1000, len(self.trajectory_history)):])
            old_features = self.attractor_features.copy()
            self.update_attractor_features(recent_trajectory)
            if old_features['center_of_mass'] is None:
                return True, 0.0
            com_change = np.linalg.norm(self.attractor_features['center_of_mass'] - old_features['center_of_mass'])
            radius_change = abs(self.attractor_features['radius_of_gyration'] - old_features['radius_of_gyration'])
            axes_change = 0
            for i in range(3):
                dot_product = np.abs(np.dot(self.attractor_features['principal_axes'][:, i],
                                           old_features['principal_axes'][:, i]))
                axes_change += 1 - dot_product
            winding_change = abs(self.attractor_features['winding_number'] - old_features['winding_number'])
            total_change = com_change + radius_change + axes_change + winding_change
            is_satisfied = total_change <= self.tolerance
            if not is_satisfied:
                self.stats['violations'] += 1
                self.stats['max_violation'] = max(self.stats['max_violation'], total_change)
                n_violations = self.stats['violations']
                self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + total_change) / n_violations
            return is_satisfied, total_change
        return True, 0.0
    def enforce_constraint(self, t, state, next_state, dt, system):
        if len(self.trajectory_history) < 100 or self.attractor_features['center_of_mass'] is None:
            return next_state
        center = self.attractor_features['center_of_mass']
        radius = self.attractor_features['radius_of_gyration']
        next_centered = next_state - center
        next_distance = np.linalg.norm(next_centered)
        if next_distance > 3 * radius:
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                direction = next_centered / next_distance
                corrected_distance = 3 * radius
                corrected_state = center + direction * corrected_distance
            elif self.enforcement_method == ConstraintEnforcementMethod.SOFT_CONSTRAINT:
                direction = next_centered / next_distance
                ideal_distance = 3 * radius
                soft_factor = 0.5
                effective_distance = next_distance + soft_factor * (ideal_distance - next_distance)
                corrected_state = center + direction * effective_distance
            else:
                direction = next_centered / next_distance
                ideal_distance = 3 * radius
                soft_factor = 0.3
                effective_distance = next_distance + soft_factor * (ideal_distance - next_distance)
                corrected_state = center + direction * effective_distance
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_attractor_features(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.trajectory_history:
            logger.warning("轨迹历史为空，无法绘图")
            return
        trajectory = np.array(self.trajectory_history)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.5)
        if self.attractor_features['center_of_mass'] is not None:
            center = self.attractor_features['center_of_mass']
            axes = self.attractor_features['principal_axes']
            radius = self.attractor_features['radius_of_gyration']
            ax.scatter([center[0]], [center[1]], [center[2]], color='r', s=100, label='中心')
            for i in range(3):
                length = radius * 2
                ax.quiver(center[0], center[1], center[2],
                         axes[0, i] * length, axes[1, i] * length, axes[2, i] * length,
                         color=['r', 'g', 'b'][i], label=f'主轴{i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz吸引子及其特征')
        ax.legend()
        plt.savefig(os.path.join(output_dir, "attractor_features.png"), dpi=300)
        plt.close()

class PhaseSpaceBoundsConstraint(PhysicalConstraint):
    def __init__(self, bounds=None, enforcement_method=ConstraintEnforcementMethod.PROJECTION):
        super().__init__("相空间边界", ConstraintType.PHASE_SPACE_BOUNDS, enforcement_method)
        default_bounds = {
            'x': (-50, 50),
            'y': (-50, 50),
            'z': (0, 100)
        }
        self.bounds = bounds or default_bounds
        self.violations_history = []
        self.time_history = []
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        x, y, z = state
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']
        z_min, z_max = self.bounds['z']
        x_violation = max(0, x_min - x, x - x_max)
        y_violation = max(0, y_min - y, y - y_max)
        z_violation = max(0, z_min - z, z - z_max)
        total_violation = x_violation + y_violation + z_violation
        self.violations_history.append(total_violation)
        self.time_history.append(t)
        is_satisfied = total_violation == 0
        if not is_satisfied:
            self.stats['violations'] += 1
            self.stats['max_violation'] = max(self.stats['max_violation'], total_violation)
            n_violations = self.stats['violations']
            self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + total_violation) / n_violations
        return is_satisfied, total_violation
    def enforce_constraint(self, t, state, next_state, dt, system):
        nx, ny, nz = next_state
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']
        z_min, z_max = self.bounds['z']
        if nx < x_min or nx > x_max or ny < y_min or ny > y_max or nz < z_min or nz > z_max:
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                corrected_x = max(x_min, min(nx, x_max))
                corrected_y = max(y_min, min(ny, y_max))
                corrected_z = max(z_min, min(nz, z_max))
                corrected_state = np.array([corrected_x, corrected_y, corrected_z])
            elif self.enforcement_method == ConstraintEnforcementMethod.SOFT_CONSTRAINT:
                ideal_x = max(x_min, min(nx, x_max))
                ideal_y = max(y_min, min(ny, y_max))
                ideal_z = max(z_min, min(nz, z_max))
                soft_factor = 0.8
                corrected_x = nx + soft_factor * (ideal_x - nx)
                corrected_y = ny + soft_factor * (ideal_y - ny)
                corrected_z = nz + soft_factor * (ideal_z - nz)
                corrected_state = np.array([corrected_x, corrected_y, corrected_z])
            elif self.enforcement_method == ConstraintEnforcementMethod.RELAXATION:
                x_violation = max(0, x_min - nx, nx - x_max)
                y_violation = max(0, y_min - ny, ny - y_max)
                z_violation = max(0, z_min - nz, nz - z_max)
                total_violation = x_violation + y_violation + z_violation
                relaxation_factor = min(1.0, total_violation / 10.0)
                ideal_x = max(x_min, min(nx, x_max))
                ideal_y = max(y_min, min(ny, y_max))
                ideal_z = max(z_min, min(nz, z_max))
                corrected_x = nx + relaxation_factor * (ideal_x - nx)
                corrected_y = ny + relaxation_factor * (ideal_y - ny)
                corrected_z = nz + relaxation_factor * (ideal_z - nz)
                corrected_state = np.array([corrected_x, corrected_y, corrected_z])
            else:
                corrected_x = max(x_min, min(nx, x_max))
                corrected_y = max(y_min, min(ny, y_max))
                corrected_z = max(z_min, min(nz, z_max))
                corrected_state = np.array([corrected_x, corrected_y, corrected_z])
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_violations_history(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.violations_history:
            logger.warning("边界违反历史为空，无法绘图")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_history, self.violations_history)
        plt.xlabel('时间')
        plt.ylabel('边界违反程度')
        plt.title('相空间边界违反历史')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "phase_space_violations.png"), dpi=300)
        plt.close()

class StatisticalInvariantsConstraint(PhysicalConstraint):
    def __init__(self, window_size=1000, enforcement_method=ConstraintEnforcementMethod.SOFT_CONSTRAINT,
                 tolerance=0.1):
        super().__init__("统计不变量", ConstraintType.STATISTICAL_INVARIANTS, enforcement_method)
        self.window_size = window_size
        self.tolerance = tolerance
        self.state_history = []
        self.time_history = []
        self.statistics = {
            'mean': None,
            'std': None,
            'skewness': None,
            'kurtosis': None,
            'correlation': None
        }
    def update_statistics(self):
        if len(self.state_history) < self.window_size:
            return
        recent_states = np.array(self.state_history[-self.window_size:])
        mean = np.mean(recent_states, axis=0)
        std = np.std(recent_states, axis=0)
        centered = recent_states - mean
        skewness = np.mean(centered**3, axis=0) / (std**3 + 1e-10)
        kurtosis = np.mean(centered**4, axis=0) / (std**4 + 1e-10) - 3
        correlation = np.corrcoef(recent_states, rowvar=False)
        self.statistics['mean'] = mean
        self.statistics['std'] = std
        self.statistics['skewness'] = skewness
        self.statistics['kurtosis'] = kurtosis
        self.statistics['correlation'] = correlation
    def check_constraint(self, t, state, system):
        super().check_constraint(t, state, system)
        self.state_history.append(state.copy())
        self.time_history.append(t)
        if len(self.state_history) < 2 * self.window_size:
            return True, 0.0
        if len(self.state_history) % (self.window_size // 10) == 0:
            old_statistics = self.statistics.copy()
            self.update_statistics()
            if old_statistics['mean'] is None:
                return True, 0.0
            mean_change = np.linalg.norm(self.statistics['mean'] - old_statistics['mean'])
            std_change = np.linalg.norm(self.statistics['std'] - old_statistics['std'])
            skewness_change = np.linalg.norm(self.statistics['skewness'] - old_statistics['skewness'])
            kurtosis_change = np.linalg.norm(self.statistics['kurtosis'] - old_statistics['kurtosis'])
            correlation_change = np.linalg.norm(self.statistics['correlation'] - old_statistics['correlation'])
            total_change = mean_change + std_change + skewness_change + kurtosis_change + correlation_change
            is_satisfied = total_change <= self.tolerance
            if not is_satisfied:
                self.stats['violations'] += 1
                self.stats['max_violation'] = max(self.stats['max_violation'], total_change)
                n_violations = self.stats['violations']
                self.stats['avg_violation'] = ((n_violations - 1) * self.stats['avg_violation'] + total_change) / n_violations
            return is_satisfied, total_change
        return True, 0.0
    def enforce_constraint(self, t, state, next_state, dt, system):
        if len(self.state_history) < self.window_size or self.statistics['mean'] is None:
            return next_state
        mean = self.statistics['mean']
        std = self.statistics['std']
        z_score = (next_state - mean) / (std + 1e-10)
        max_z_score = np.max(np.abs(z_score))
        if max_z_score > 3.0:
            if self.enforcement_method == ConstraintEnforcementMethod.PROJECTION:
                limited_z_score = np.clip(z_score, -3.0, 3.0)
                corrected_state = mean + limited_z_score * std
            elif self.enforcement_method == ConstraintEnforcementMethod.SOFT_CONSTRAINT:
                ideal_z_score = np.clip(z_score, -3.0, 3.0)
                soft_factor = 0.5
                effective_z_score = z_score + soft_factor * (ideal_z_score - z_score)
                corrected_state = mean + effective_z_score * std
            else:
                ideal_z_score = np.clip(z_score, -3.0, 3.0)
                soft_factor = 0.3
                effective_z_score = z_score + soft_factor * (ideal_z_score - z_score)
                corrected_state = mean + effective_z_score * std
            self.stats['enforcements'] += 1
            return corrected_state
        return next_state
    def plot_statistics_history(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.state_history:
            logger.warning("状态历史为空，无法绘图")
            return
        states = np.array(self.state_history)
        times = np.array(self.time_history)
        window_size = min(self.window_size, len(states) // 10)
        if window_size < 10:
            logger.warning("数据点太少，无法计算滑动窗口统计量")
            return
        n_points = len(states) - window_size + 1
        window_means = np.zeros((n_points, 3))
        window_stds = np.zeros((n_points, 3))
        window_times = times[window_size-1:]
        for i in range(n_points):
            window = states[i:i+window_size]
            window_means[i] = np.mean(window, axis=0)
            window_stds[i] = np.std(window, axis=0)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(window_times, window_means[:, 0])
        plt.ylabel('X均值')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(window_times, window_means[:, 1])
        plt.ylabel('Y均值')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(window_times, window_means[:, 2])
        plt.xlabel('时间')
        plt.ylabel('Z均值')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_history.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(window_times, window_stds[:, 0])
        plt.ylabel('X标准差')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(window_times, window_stds[:, 1])
        plt.ylabel('Y标准差')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(window_times, window_stds[:, 2])
        plt.xlabel('时间')
        plt.ylabel('Z标准差')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "std_history.png"), dpi=300)
        plt.close()

class PhysicalConstraintsManager:
    def __init__(self):
        self.constraints = {}
        self.check_times = []
        self.enforce_times = []
    def add_constraint(self, constraint):
        self.constraints[constraint.name] = constraint
        logger.info(f"添加约束: {constraint.name}")
    def remove_constraint(self, name):
        if name in self.constraints:
            del self.constraints[name]
            logger.info(f"移除约束: {name}")
    def check_constraints(self, t, state, system):
        start_time = time.time()
        violations = []
        for name, constraint in self.constraints.items():
            is_satisfied, violation = constraint.check_constraint(t, state, system)
            if not is_satisfied:
                violations.append((name, violation))
        self.check_times.append(time.time() - start_time)
        return violations
    def enforce_constraints(self, t, state, next_state, dt, system):
        start_time = time.time()
        corrected_state = next_state.copy()
        priority_order = [
            ConstraintType.PHASE_SPACE_BOUNDS,
            ConstraintType.ENERGY,
            ConstraintType.VOLUME_CONTRACTION,
            ConstraintType.DISSIPATION,
            ConstraintType.ATTRACTOR_TOPOLOGY,
            ConstraintType.STATISTICAL_INVARIANTS
        ]
        for constraint_type in priority_order:
            for name, constraint in self.constraints.items():
                if constraint.constraint_type == constraint_type:
                    corrected_state = constraint.enforce_constraint(t, state, corrected_state, dt, system)
        self.enforce_times.append(time.time() - start_time)
        return corrected_state
    def get_stats_report(self):
        report = "物理约束管理器统计报告:\n\n"
        for name, constraint in self.constraints.items():
            report += constraint.get_stats_report() + "\n"
        if self.check_times:
            avg_check_time = sum(self.check_times) / len(self.check_times)
            max_check_time = max(self.check_times)
            report += "约束检查性能:\n"
            report += f"平均检查时间: {avg_check_time:.6f}秒\n"
            report += f"最大检查时间: {max_check_time:.6f}秒\n"
            report += f"总检查次数: {len(self.check_times)}\n\n"
        if self.enforce_times:
            avg_enforce_time = sum(self.enforce_times) / len(self.enforce_times)
            max_enforce_time = max(self.enforce_times)
            report += "约束实施性能:\n"
            report += f"平均实施时间: {avg_enforce_time:.6f}秒\n"
            report += f"最大实施时间: {max_enforce_time:.6f}秒\n"
            report += f"总实施次数: {len(self.enforce_times)}\n"
        return report
    def reset_stats(self):
        for constraint in self.constraints.values():
            constraint.reset_stats()
        self.check_times = []
        self.enforce_times = []
    def plot_all_constraints(self, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for name, constraint in self.constraints.items():
            if hasattr(constraint, 'plot_energy_history'):
                constraint.plot_energy_history(output_dir)
            if hasattr(constraint, 'plot_contraction_history'):
                constraint.plot_contraction_history(output_dir)
            if hasattr(constraint, 'plot_dissipation_history'):
                constraint.plot_dissipation_history(output_dir)
            if hasattr(constraint, 'plot_attractor_features'):
                constraint.plot_attractor_features(output_dir)
            if hasattr(constraint, 'plot_violations_history'):
                constraint.plot_violations_history(output_dir)
            if hasattr(constraint, 'plot_statistics_history'):
                constraint.plot_statistics_history(output_dir)

class PhysicalConstraintsIntegrator:
    def __init__(self, system, base_solver, constraints_manager=None):
        self.system = system
        self.base_solver = base_solver
        self.constraints_manager = constraints_manager or PhysicalConstraintsManager()
        self.step_times = []
    def add_constraint(self, constraint):
        self.constraints_manager.add_constraint(constraint)
    def step(self, t, state, dt):
        start_time = time.time()
        violations = self.constraints_manager.check_constraints(t, state, self.system)
        next_state = self.base_solver.step(t, state, dt)
        corrected_state = self.constraints_manager.enforce_constraints(t, state, next_state, dt, self.system)
        self.step_times.append(time.time() - start_time)
        return corrected_state
    def solve(self, t_span, initial_state, dt):
        t_start, t_end = t_span
        t_current = t_start
        state_current = np.array(initial_state)
        t_points = [t_current]
        states = [state_current.copy()]
        self.constraints_manager.reset_stats()
        while t_current < t_end:
            next_state = self.step(t_current, state_current, dt)
            t_current += dt
            state_current = next_state
            t_points.append(t_current)
            states.append(state_current.copy())
        t_array = np.array(t_points)
        states_array = np.array(states)
        return t_array, states_array
    def get_performance_report(self):
        report = "物理约束积分器性能报告:\n\n"
        report += self.constraints_manager.get_stats_report() + "\n"
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            max_step_time = max(self.step_times)
            min_step_time = min(self.step_times)
            report += "积分性能:\n"
            report += f"平均步骤时间: {avg_step_time:.6f}秒\n"
            report += f"最大步骤时间: {max_step_time:.6f}秒\n"
            report += f"最小步骤时间: {min_step_time:.6f}秒\n"
            report += f"总步数: {len(self.step_times)}\n"
            report += f"总计算时间: {sum(self.step_times):.6f}秒\n"
        return report
    def plot_results(self, t, states, output_dir="physical_constraints_plots"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz系统轨迹（物理约束）')
        plt.savefig(os.path.join(output_dir, "lorenz_trajectory_3d.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, states[:, 0])
        plt.ylabel('X')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(t, states[:, 1])
        plt.ylabel('Y')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(t, states[:, 2])
        plt.xlabel('时间')
        plt.ylabel('Z')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lorenz_coordinates.png"), dpi=300)
        plt.close()
        self.constraints_manager.plot_all_constraints(output_dir)

if __name__ == "__main__":
    from lorenz_system import LorenzSystem
    from numerical_solvers import RK4Solver
    system = LorenzSystem()
    base_solver = RK4Solver(system)
    constraints_manager = PhysicalConstraintsManager()
    energy_constraint = EnergyConstraint(
        enforcement_method=ConstraintEnforcementMethod.PROJECTION,
        tolerance=1e-3,
        strict=False
    )
    constraints_manager.add_constraint(energy_constraint)
    volume_constraint = VolumeContractionConstraint(
        enforcement_method=ConstraintEnforcementMethod.PROJECTION,
        tolerance=1e-3
    )
    constraints_manager.add_constraint(volume_constraint)
    dissipation_constraint = DissipationConstraint(
        enforcement_method=ConstraintEnforcementMethod.SOFT_CONSTRAINT,
        tolerance=1e-3
    )
    constraints_manager.add_constraint(dissipation_constraint)
    bounds_constraint = PhaseSpaceBoundsConstraint(
        enforcement_method=ConstraintEnforcementMethod.PROJECTION
    )
    constraints_manager.add_constraint(bounds_constraint)
    integrator = PhysicalConstraintsIntegrator(system, base_solver, constraints_manager)
    t_span = [0, 100]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    t, states = integrator.solve(t_span, initial_state, dt)
    print(integrator.get_performance_report())
    integrator.plot_results(t, states)
