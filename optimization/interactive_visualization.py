import numpy as np
import warnings
import functools
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lorenz_system_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LorenzSystem")

def safe_divide(a, b, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress runtime warnings for division by zero / invalid ops
        if np.isscalar(b):
            return a / b if b != 0 else default
        else:
            result = np.divide(a, b, out=np.full_like(a, default, dtype=np.result_type(a,b)), where=b!=0)
            # Handle potential NaNs resulting from 0/0 if default isn't NaN
            if not np.isnan(default):
                 result[np.isnan(result)] = default
            return result


def safe_log(x, epsilon=1e-10):
     if np.isscalar(x):
         return np.log(max(x, epsilon))
     else:
         x_safe = np.maximum(x, epsilon)
         return np.log(x_safe)


def safe_sqrt(x, epsilon=0.0):
     if np.isscalar(x):
         return np.sqrt(max(x, epsilon))
     else:
         x_safe = np.maximum(x, epsilon)
         return np.sqrt(x_safe)


def check_numerical_stability(array, name="array"):
    if not isinstance(array, np.ndarray):
         # If not an array, assume stable for now or convert if possible
         try:
             array = np.array(array)
         except Exception:
             logger.warning(f"Input '{name}' is not array-like and could not be checked for stability.")
             return True # Assume stable if cannot check

    if np.any(np.isnan(array)):
        logger.warning(f"NaN值出现在{name}中")
        return False
    if np.any(np.isinf(array)):
        logger.warning(f"无穷值出现在{name}中")
        return False
    # Check against a reasonable large number, not necessarily fixed at 1e10
    # Use finfo for float limits if appropriate, or a configurable threshold
    float_limit = np.finfo(array.dtype).max / 10 if np.issubdtype(array.dtype, np.floating) else 1e12
    if np.any(np.abs(array) > float_limit):
         logger.warning(f"极大值 (>{float_limit:.2E}) 出现在{name}中: max abs={np.max(np.abs(array)):.2E}")
         return False

    return True

def stabilize_array(array, max_value=None, replace_value=0.0):
     result = array.copy() # Work on a copy

     # Determine max_value based on dtype if not provided
     if max_value is None:
         max_value = np.finfo(result.dtype).max / 10 if np.issubdtype(result.dtype, np.floating) else 1e12


     # Replace NaN and Inf first
     nan_mask = np.isnan(result)
     inf_mask = np.isinf(result)
     result[nan_mask | inf_mask] = replace_value

     # Clip large values
     overflow_mask = np.abs(result) > max_value
     # Use np.clip for potentially faster operation if replacing with bounds
     # result = np.clip(result, -max_value, max_value)
     # Or replace with replace_value if that's the desired behavior:
     result[overflow_mask] = replace_value # Or np.sign(result[overflow_mask]) * max_value


     if np.any(nan_mask): logger.debug(f"Stabilized NaNs in array.")
     if np.any(inf_mask): logger.debug(f"Stabilized Infs in array.")
     if np.any(overflow_mask): logger.debug(f"Stabilized overflows (> {max_value:.2E}) in array.")


     return result

def numerical_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
             # Optionally check input arguments stability
             for i, arg in enumerate(args):
                  if isinstance(arg, np.ndarray):
                       if not check_numerical_stability(arg, f"Input arg {i} to {func_name}"):
                            logger.warning(f"Unstable input detected for {func_name}. Proceeding cautiously.")
                            # Potentially stabilize input or return early? Depends on desired behavior.
                            # args = list(args)
                            # args[i] = stabilize_array(arg)
                            # args = tuple(args)


             # Execute function with warning capture
             with warnings.catch_warnings(record=True) as caught_warnings:
                 warnings.simplefilter("always") # Catch all warnings
                 result = func(*args, **kwargs)
                 if caught_warnings:
                     for w in caught_warnings:
                         logger.warning(f"{func_name} generated warning: {w.category.__name__} - {w.message}")

             # Check result stability
             if isinstance(result, np.ndarray):
                 if not check_numerical_stability(result, f"Result of {func_name}"):
                      logger.warning(f"{func_name} returned unstable array. Stabilizing...")
                      result = stabilize_array(result)
             elif isinstance(result, (float, int)) and not np.isfinite(result):
                  logger.warning(f"{func_name} returned non-finite scalar ({result}). Replacing with 0.")
                  result = 0.0 # Default for non-finite scalar


             return result

        except FloatingPointError as fpe: # Catch specific numpy errors
            logger.error(f"{func_name} encountered FloatingPointError: {fpe}")
            # Attempt to return a default based on annotations or guess
        except ValueError as ve: # Catch potential value errors (e.g., math domain)
            logger.error(f"{func_name} encountered ValueError: {ve}")
        except TypeError as te: # Catch type errors
            logger.error(f"{func_name} encountered TypeError: {te}")
        except Exception as e: # Catch any other exceptions
            logger.error(f"{func_name} failed with unexpected error: {type(e).__name__} - {e}", exc_info=True) # Log traceback


        # Default return value logic on error
        logger.info(f"Returning default value from {func_name} due to error.")
        # Try to infer a sensible default based on return type annotation
        try:
            return_type = func.__annotations__.get('return', None)
            if return_type == np.ndarray:
                # Try finding an input array to match shape, otherwise default Lorenz state
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        return np.zeros_like(arg)
                logger.warning(f"{func_name}: Could not infer array shape, returning default [0,0,0].")
                return np.array([0.0, 0.0, 0.0])
            elif return_type == float: return 0.0
            elif return_type == int: return 0
            elif return_type == bool: return False
            else: return None # Default None if type unknown/unhandled
        except Exception:
            return None # Fallback if annotation check fails


    return wrapper

class NumericalStabilityMonitor:
    def __init__(self, log_dir="stability_logs", log_level=logging.WARNING):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "stability_log.txt")

        # Setup specific logger for stability issues
        self.stability_logger = logging.getLogger("StabilityMonitor")
        self.stability_logger.setLevel(log_level)
        # Prevent propagating to root logger if specific handling is desired
        self.stability_logger.propagate = False
        # Add handler if not already present
        if not self.stability_logger.handlers:
            fh = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.stability_logger.addHandler(fh)
            # Optionally add stream handler for console output
            # sh = logging.StreamHandler()
            # sh.setFormatter(formatter)
            # self.stability_logger.addHandler(sh)

        self.reset_stats()


    
    def check_state(self, t, state, name="state"):
        self.stats["total_checks"] += 1
        is_stable = True
        issues = []
        
        if not isinstance(state, np.ndarray):
             try:
                 state = np.array(state) # Try conversion
             except Exception:
                  self.stability_logger.warning(f"T={t:.4f}: Input '{name}' is not array-like. Cannot check stability.")
                  return True # Cannot check, assume stable


        # Check for NaN
        nan_mask = np.isnan(state)
        if np.any(nan_mask):
            self.stats["nan_count"] += 1
            is_stable = False
            issues.append(f"NaN at indices {np.where(nan_mask)[0]}")

        # Check for Inf
        inf_mask = np.isinf(state)
        if np.any(inf_mask):
            self.stats["inf_count"] += 1
            is_stable = False
            issues.append(f"Inf at indices {np.where(inf_mask)[0]}")

        # Check for Overflow (using float limits)
        if np.issubdtype(state.dtype, np.floating):
            limit = np.finfo(state.dtype).max / 10
            overflow_mask = np.abs(state) > limit
            if np.any(overflow_mask):
                self.stats["overflow_count"] += 1
                is_stable = False
                max_abs_val = np.max(np.abs(state[overflow_mask]))
                issues.append(f"Overflow (> {limit:.2E}) at indices {np.where(overflow_mask)[0]}, max abs val {max_abs_val:.2E}")
        
        if not is_stable:
            self.stats["unstable_steps"] += 1
            # Log using the dedicated stability logger
            self.stability_logger.warning(f"T={t:.6f}, Unstable '{name}': {'; '.join(issues)}. Values: {state}")
        
        return is_stable

    
    def stabilize_state(self, state, prev_state=None, max_value=None, replace_value=0.0):
         result = state.copy()

         if max_value is None:
             max_value = np.finfo(result.dtype).max / 10 if np.issubdtype(result.dtype, np.floating) else 1e12

         nan_mask = np.isnan(result)
         inf_mask = np.isinf(result)
         overflow_mask = np.abs(result) > max_value

         instability_mask = nan_mask | inf_mask | overflow_mask

         if np.any(instability_mask):
             self.stability_logger.info(f"Stabilizing state. Issues found: NaN={np.any(nan_mask)}, Inf={np.any(inf_mask)}, Overflow={np.any(overflow_mask)}")
             if prev_state is not None and prev_state.shape == result.shape:
                 # Replace only unstable values with corresponding previous state value
                 result[instability_mask] = prev_state[instability_mask]
                 # Check if prev_state itself had issues at those indices
                 if not np.all(np.isfinite(result[instability_mask])):
                      self.stability_logger.warning("Previous state also had issues at unstable indices. Replacing with {replace_value}.")
                      result[instability_mask & ~np.isfinite(result)] = replace_value
             else:
                 # If no previous state or shape mismatch, replace with default value
                 if prev_state is not None:
                      self.stability_logger.warning("Previous state unusable for stabilization (missing or shape mismatch).")
                 result[instability_mask] = replace_value
         return result

    
    def get_stability_report(self):
        total = self.stats["total_checks"]
        if total == 0:
            return "尚未进行稳定性检查。"
        
        unstable_percent = (self.stats["unstable_steps"] / total) * 100 if total > 0 else 0
        
        report = f"--- 稳定性监控报告 ---\n"
        report += f"总检查次数: {total}\n"
        report += f"不稳定步骤数: {self.stats['unstable_steps']} ({unstable_percent:.2f}%)\n"
        report += f"  - NaN 出现次数: {self.stats['nan_count']}\n"
        report += f"  - Inf 出现次数: {self.stats['inf_count']}\n"
        report += f"  - 溢出 (> 1e+N) 出现次数: {self.stats['overflow_count']}\n"
        report += f"详细日志文件: {os.path.abspath(self.log_file)}\n"
        report += "-------------------------"
        return report
    
    def reset_stats(self):
        self.stats = {
            "nan_count": 0,
            "inf_count": 0,
            "overflow_count": 0,
            "total_checks": 0,
            "unstable_steps": 0
        }
        # Optionally clear the log file on reset
        # try:
        #     with open(self.log_file, "w") as f:
        #         f.write(f"Stability log reset at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # except Exception as e:
        #     self.stability_logger.error(f"Could not clear log file {self.log_file}: {e}")




def enhance_lorenz_system(lorenz_system_class):
    class EnhancedLorenzSystem(lorenz_system_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Use a shared monitor or create one per instance? Instance-specific seems better.
            self.stability_monitor = NumericalStabilityMonitor(log_dir=f"logs_{self.__class__.__name__}")
            self.last_valid_state = None # Store last known good state for recovery

        @numerical_error_handler # Apply decorator
        def derivatives(self, t, state):
             # 1. Check input state stability
             if not self.stability_monitor.check_state(t, state, "Derivatives Input"):
                  logger.warning(f"T={t:.4f}: Unstable input state received by derivatives.")
                  # Option 1: Stabilize input using last valid state or default
                  if self.last_valid_state is not None and self.last_valid_state.shape == state.shape:
                       stabilized_input = self.stability_monitor.stabilize_state(state, self.last_valid_state)
                       logger.info(f"T={t:.4f}: Stabilized derivatives input from {state} to {stabilized_input}")
                       state = stabilized_input
                  else:
                       # Cannot recover reliably, maybe raise error or return zeros?
                       logger.error(f"T={t:.4f}: Cannot stabilize unstable input state, returning zeros.")
                       return np.zeros_like(state)


             # 2. Calculate derivatives using the (potentially stabilized) state
             dxdt = super().derivatives(t, state) # Call original method


             # 3. Store current state as last valid state *before* returning derivatives
             # This assumes the input `state` was valid or successfully stabilized
             self.last_valid_state = state.copy()


             # 4. Check output stability (decorator already does this, but can add specific logging)
             if not np.all(np.isfinite(dxdt)):
                  logger.warning(f"T={t:.4f}: Original derivatives calculation produced non-finite values: {dxdt}")
                  # Decorator will handle stabilization if check_numerical_stability finds issue

             return dxdt # Decorator handles final stability check and stabilization

        @numerical_error_handler # Apply decorator
        def pseudo_energy(self, state):
             # Input check (optional, depends if called directly with potentially bad state)
             if not self.stability_monitor.check_state(0.0, state, "PseudoEnergy Input"): # Use t=0 as placeholder
                  logger.warning("Unstable state provided to pseudo_energy.")
                  # Return NaN or default? Let decorator handle scalar result.
                  # return np.nan # Indicate error state explicitly


             x, y, z = state
             # Use safe operations
             term1 = safe_divide(x**2, 2.0 * self.sigma) # Assuming original formula involved sigma? Adjust as needed.
             term2 = safe_divide(y**2, 2.0)
             term3 = safe_divide((z - self.rho - self.sigma)**2, 2.0) # Example adjustment
             
             energy = term1 + term2 - term3 # Example formula
             
             # Decorator will check scalar result
             return energy
        
        def get_stability_report(self):
            return self.stability_monitor.get_stability_report()

        def reset_monitor(self):
            self.stability_monitor.reset_stats()
            self.last_valid_state = None


    return EnhancedLorenzSystem

def enhance_numerical_solver(solver_class):
    class EnhancedNumericalSolver(solver_class):
        def __init__(self, system, *args, **kwargs): # Ensure system is passed
            # Ensure the system passed is also enhanced, if possible
            if not isinstance(system, NumericalStabilityMonitor): # Check if system has monitor
                logger.warning(f"Solver {self.__class__.__name__} initialized with a non-enhanced system {type(system).__name__}. Stability monitoring might be limited.")

            super().__init__(system, *args, **kwargs) # Pass system to parent
            self.stability_monitor = NumericalStabilityMonitor(log_dir=f"logs_{self.__class__.__name__}")
            self.last_valid_state = None
            self.min_dt = kwargs.get('min_dt', 1e-8) # Minimum allowed timestep
            self.max_dt_reductions = kwargs.get('max_dt_reductions', 5) # Max times dt is halved


        # Decorator applied here will catch errors within the step method itself
        @numerical_error_handler
        def step(self, t, state, dt):
             current_dt = dt
             attempt = 0
             max_attempts = self.max_dt_reductions + 1 # Allow N reductions + 1 initial try

             while attempt < max_attempts:
                 try:
                     # Check input state stability before attempting step
                     if not self.stability_monitor.check_state(t, state, f"{self.__class__.__name__} Step Input"):
                         logger.error(f"T={t:.4f}: Unstable state provided to step method. Cannot proceed.")
                         # Return previous state or error state? Returning current state might lead to loop.
                         # Perhaps return a special marker or raise? For now, return copy.
                         return state.copy()

                     self.last_valid_state = state.copy() # Store last known good state *before* step

                     # Execute the original step method
                     next_state = super().step(t, state, current_dt)

                     # Check result stability
                     if self.stability_monitor.check_state(t + current_dt, next_state, f"{self.__class__.__name__} Step Result"):
                         # Stable result, return it
                         return next_state
                     else:
                         # Result is unstable
                         logger.warning(f"T={t + current_dt:.4f}: Step produced unstable result (attempt {attempt + 1}/{max_attempts}).")
                         if attempt < max_attempts -1 :
                             current_dt /= 2.0 # Reduce timestep
                             if current_dt < self.min_dt:
                                  logger.error(f"T={t:.4f}: Timestep reduced below minimum ({self.min_dt:.2E}). Stabilizing and returning.")
                                  return self.stability_monitor.stabilize_state(next_state, self.last_valid_state)
                             logger.info(f"Reducing timestep to {current_dt:.2E} and retrying.")
                             # state remains the same for the retry with smaller dt
                         else:
                             logger.error(f"T={t:.4f}: Max timestep reduction attempts reached. Stabilizing unstable result.")
                             return self.stability_monitor.stabilize_state(next_state, self.last_valid_state)

                 except Exception as e:
                     logger.error(f"T={t:.4f}: Error during solver step (attempt {attempt + 1}/{max_attempts}): {e}", exc_info=True)
                     # Option: Try recovery (e.g., Euler), stabilize, or return last valid state
                     logger.warning(f"Attempting recovery by stabilizing last valid state.")
                     return self.last_valid_state # Return last known good state on error

                 attempt += 1

             # Should not be reached if logic is correct, but as fallback:
             logger.error(f"T={t:.4f}: Exited step retry loop unexpectedly. Returning last valid state.")
             return self.last_valid_state


        # The solve method orchestrates steps, handles overall progress and stability logging
        # No numerical_error_handler decorator needed here as step handles its errors.
        def solve(self, t_span, initial_state, dt):
             t_start, t_end = t_span
             t_current = t_start
             state_current = np.array(initial_state)
             base_dt = dt # Store the initial dt

             t_points = [t_current]
             states = [state_current.copy()]
             self.stability_monitor.reset_stats() # Reset for this solve run
             self.last_valid_state = state_current.copy() # Initialize last valid state

             max_simulation_steps = int((t_end - t_start) / self.min_dt) * 2 # Safety break
             step_count = 0

             while t_current < t_end and step_count < max_simulation_steps:
                  # Determine current dt (can be adaptive based on previous step, etc.)
                  # For now, just use base_dt, as step handles internal reduction
                  current_dt = min(base_dt, t_end - t_current) # Ensure not overstepping t_end

                  if current_dt < self.min_dt / 10: # Avoid extremely small final steps
                       break

                  # Execute one step (which includes stability checks and retries)
                  next_state = self.step(t_current, state_current, current_dt)

                  # Update state and time
                  # The actual time increment might be smaller if step reduced dt internally,
                  # but for simplicity, we advance by the *requested* dt here.
                  # A more advanced adaptive solver would return the actual dt used.
                  t_current += current_dt
                  state_current = next_state

                  # Store results
                  t_points.append(t_current)
                  states.append(state_current.copy())
                  self.last_valid_state = state_current.copy() # Update after successful step

                  step_count += 1

             if step_count >= max_simulation_steps:
                  logger.warning(f"Solve terminated: Maximum simulation steps ({max_simulation_steps}) reached.")
             elif t_current < t_end:
                  logger.warning(f"Solve terminated early at T={t_current:.4f} (before T_end={t_end:.4f}).")


             stability_report = self.stability_monitor.get_stability_report()
             logger.info(f"Solve completed. {stability_report}")

             return np.array(t_points), np.array(states)
        
        def get_stability_report(self):
            return self.stability_monitor.get_stability_report()

        def reset_monitor(self):
            self.stability_monitor.reset_stats()
            self.last_valid_state = None


    return EnhancedNumericalSolver


if __name__ == "__main__":
    try:
         from lorenz_system import LorenzSystem
         from numerical_solvers import RK4Solver
    except ImportError:
         print("Could not import LorenzSystem or RK4Solver. Using placeholder classes.")
         # Define minimal placeholders if imports fail
         class LorenzSystem:
             def __init__(self, sigma=10, rho=28, beta=8/3): pass
             def derivatives(self, t, state): return np.zeros_like(state)
         class RK4Solver:
             def __init__(self, system): self.system = system
             def step(self, t, state, dt):
                 # Simulate potential instability for testing
                 if np.random.rand() < 0.01: # Small chance of failure
                     # return state + np.random.randn(3) * 1e12 # Overflow
                     return np.array([np.nan, np.inf, 0]) # NaN/Inf
                 return state + dt * 0.1 # Dummy step
             def solve(self, t_span, initial_state, dt): # Basic solve for placeholder
                 t = np.arange(t_span[0], t_span[1]+dt, dt)
                 states = np.zeros((len(t), len(initial_state)))
                 states[0] = initial_state
                 for i in range(len(t)-1):
                     states[i+1] = self.step(t[i], states[i], dt)
                 return t, states


    
    EnhancedLorenzSystem = enhance_lorenz_system(LorenzSystem)
    EnhancedRK4Solver = enhance_numerical_solver(RK4Solver)
    
    system = EnhancedLorenzSystem()
    solver = EnhancedRK4Solver(system, min_dt=1e-6) # Pass system, set min_dt
    
    t_span = [0, 50] # Shorter span for example
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.02
    
    print(f"Running enhanced solver for t_span={t_span} with dt={dt}...")
    t, states = solver.solve(t_span, initial_state, dt)
    
    print("\n--- Final Stability Report ---")
    print(solver.get_stability_report())

    # Basic plot to visualize result
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[:, 0], states[:, 1], states[:, 2])
        ax.set_title("Enhanced Solver Trajectory")
        plt.show()
    except ImportError:
        print("Matplotlib not found. Cannot plot trajectory.")
    except Exception as e:
        print(f"Plotting failed: {e}")
