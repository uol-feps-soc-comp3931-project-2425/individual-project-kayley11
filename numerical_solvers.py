
import numpy as np
from lorenz_system import LorenzSystem

class NumericalSolver:

    def __init__(self, system=None):

        self.system = system if system is not None else LorenzSystem()
        self.name = "Base Solver"
    
    def step(self, t, state, dt):

        raise NotImplementedError("子类必须实现step方法")
    
    def solve(self, t_span, initial_state, dt):

        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt/2, dt)
        n_steps = len(t)
        

        states = np.zeros((n_steps, len(initial_state)))
        states[0] = initial_state
        

        for i in range(1, n_steps):
            states[i] = self.step(t[i-1], states[i-1], dt)
        
        return t, states


class EulerSolver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "Euler Method"
    
    def step(self, t, state, dt):

        derivatives = self.system.derivatives(t, state)
        return state + dt * derivatives


class MidpointSolver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "Midpoint Method"
    
    def step(self, t, state, dt):

        k1 = self.system.derivatives(t, state)
        mid_state = state + 0.5 * dt * k1
        
        k2 = self.system.derivatives(t + 0.5 * dt, mid_state)
        return state + dt * k2


class RK2Solver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "RK2 Method"
    
    def step(self, t, state, dt):

        k1 = self.system.derivatives(t, state)
        k2 = self.system.derivatives(t + dt, state + dt * k1)
        return state + dt * (0.5 * k1 + 0.5 * k2)


class RK4Solver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "RK4 Method"
    
    def step(self, t, state, dt):

        k1 = self.system.derivatives(t, state)
        k2 = self.system.derivatives(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = self.system.derivatives(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = self.system.derivatives(t + dt, state + dt * k3)
        
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class SymplecticEulerSolver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "Symplectic Euler Method"
    
    def step(self, t, state, dt):

        x, y, z = state
        sigma, rho, beta = self.system.sigma, self.system.rho, self.system.beta
        

        x_half = x + 0.5 * dt * sigma * (y - x)
        

        y_new = y + dt * (x_half * (rho - z) - y)
        z_new = z + dt * (x_half * y - beta * z)
        

        x_new = x_half + 0.5 * dt * sigma * (y_new - x_half)
        
        return np.array([x_new, y_new, z_new])


class EnergyPreservingSolver(NumericalSolver):

    def __init__(self, system=None):
        super().__init__(system)
        self.name = "Energy Preserving Method"
    
    def step(self, t, state, dt):


        rk4_solver = RK4Solver(self.system)
        predicted_state = rk4_solver.step(t, state, dt)
        
        initial_energy = self.system.pseudo_energy(state)
        

        predicted_energy = self.system.pseudo_energy(predicted_state)

        volume_factor = np.exp(self.system.volume_contraction_rate * dt)
        

        target_energy = initial_energy * volume_factor

        if abs(predicted_energy - target_energy) > 1e-6 * target_energy:

            scale_factor = np.sqrt(target_energy / predicted_energy)
            
        
            x_new, y_new, z_new = predicted_state
            x_new *= scale_factor
            y_new *= scale_factor
            
            return np.array([x_new, y_new, z_new])
        else:
            return predicted_state
