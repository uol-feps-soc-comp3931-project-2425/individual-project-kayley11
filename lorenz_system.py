
import numpy as np

class LorenzSystem:
    """
    Lorenz系统类，包含基本方程和参数
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        
        self.volume_contraction_rate = -(self.sigma + 1 + self.beta)
        
    def derivatives(self, t, state):
        
        x, y, z = state
        
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def pseudo_energy(self, state):
        
        x, y, z = state

        return 0.5 * (x**2 + y**2 + (z - self.rho)**2)
    
    def volume_contraction(self, state=None):

        return self.volume_contraction_rate
