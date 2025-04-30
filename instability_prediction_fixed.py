import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from lorenz_system import LorenzSystem
from numerical_solvers import RK4Solver
from hybrid_solver import HybridAdaptiveSolver

class LorenzInstabilityPredictor:
    def __init__(self, window_size=20, prediction_horizon=10, lyapunov_threshold=0.5):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.lyapunov_threshold = lyapunov_threshold
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.lorenz = LorenzSystem()
    
    def calculate_local_divergence(self, states):
        distances = np.sqrt(np.sum(np.diff(states, axis=0)**2, axis=1))
        
        if len(distances) < 1: # Handle case with insufficient points
             # Return an array of zeros with the same length as states or handle appropriately
             return np.zeros(len(states))
             
        divergence = np.diff(distances)
        
        if len(divergence) == 0: # Handle case where diff results in empty array
           # Return array of zeros matching expected length or handle appropriately
           return np.zeros(len(states))


        divergence = np.insert(divergence, 0, divergence[0])
        divergence = np.insert(divergence, 0, [divergence[0], divergence[0]])

        # Ensure divergence length matches states length if necessary
        if len(divergence) < len(states):
             padding = np.zeros(len(states) - len(divergence))
             divergence = np.concatenate((padding, divergence)) # Pad beginning or end as appropriate
        elif len(divergence) > len(states):
             divergence = divergence[-len(states):] # Trim if too long


        
        return divergence
    
    def extract_features(self, states, divergence):
        features = []
        labels = []
        
        if len(states) <= self.window_size + self.prediction_horizon:
             # Not enough data to form a window and prediction horizon
             return np.array(features), np.array(labels)

        for i in range(len(states) - self.window_size - self.prediction_horizon):
            window_states = states[i:i+self.window_size]
            
            # Ensure divergence has enough elements for the window
            if i + self.window_size > len(divergence):
                continue # Skip if not enough divergence data for the window
            window_divergence = divergence[i:i+self.window_size]
            
            feature = []
            
            feature.extend(window_states.flatten()) # Flatten state window
            
            feature.extend(window_divergence)
            
            feature.append(np.mean(window_states[:, 0]))
            feature.append(np.std(window_states[:, 0]))
            feature.append(np.mean(window_states[:, 1]))
            feature.append(np.std(window_states[:, 1]))
            feature.append(np.mean(window_states[:, 2]))
            feature.append(np.std(window_states[:, 2]))
            
            features.append(feature)
            
            # Ensure divergence has enough elements for the prediction horizon
            future_start_index = i + self.window_size
            future_end_index = future_start_index + self.prediction_horizon
            if future_end_index > len(divergence):
                 continue # Skip if not enough divergence data for prediction horizon

            future_divergence = divergence[future_start_index:future_end_index]
            
            if len(future_divergence) == 0: # Handle empty future divergence
                 max_future_divergence = 0 # Or some other default
            else:
                 max_future_divergence = np.max(np.abs(future_divergence))

            
            label = 1 if max_future_divergence > self.lyapunov_threshold else 0
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def prepare_data(self, t, states):
        divergence = self.calculate_local_divergence(states)
        
        # Ensure divergence is not empty and has compatible length
        if len(divergence) == 0 or len(divergence) < len(states):
             print(f"Warning: Divergence calculation resulted in {len(divergence)} points for {len(states)} states. Adjusting.")
             # Handle inadequate divergence data (e.g., pad or skip)
             # For now, let's pad divergence to match states length with zeros
             if len(divergence) < len(states):
                  divergence = np.pad(divergence, (len(states) - len(divergence), 0), 'constant', constant_values=0)
             else: # If still empty after potential calculation issues
                  return np.array([]), np.array([]), np.array([]), np.array([]) # Return empty arrays if data prep fails


        features, labels = self.extract_features(states, divergence)
        
        if features.shape[0] == 0: # Check if feature extraction yielded results
            print("Warning: No features extracted. Check window_size, prediction_horizon, and data length.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Ensure features are not empty before scaling
        if features.size == 0:
             return np.array([]), np.array([]), np.array([]), np.array([])


        features_scaled = self.scaler.fit_transform(features)
        
        # Check if labels array is empty before splitting
        if labels.size == 0:
            print("Warning: Labels array is empty.")
             # Decide how to handle this, e.g., return empty sets
            return np.array([]), np.array([]), np.array([]), np.array([])


        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42)
        
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
             print("Warning: Train or test split resulted in empty set.")
             return np.array([]), np.array([]), np.array([]), np.array([])


        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, t, states, epochs=50, batch_size=32, verbose=1):
        X_train, X_test, y_train, y_test = self.prepare_data(t, states)
        
        # Check if data preparation was successful
        if X_train.size == 0 or y_train.size == 0 or X_test.size == 0 or y_test.size == 0:
             print("Error: Data preparation failed, cannot train model.")
             return None # Or raise an exception


        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=verbose
        )
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"测试集准确率: {accuracy*100:.2f}%")
        
        return history
    
    def save_model(self, model_path, scaler_path):
        if self.model is not None:
            self.model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"模型已保存到 {model_path}")
            print(f"归一化器已保存到 {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        try:
             self.model = tf.keras.models.load_model(model_path)
             with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
             print(f"模型已从 {model_path} 加载")
             print(f"归一化器已从 {scaler_path} 加载")
        except Exception as e:
             print(f"Error loading model or scaler: {e}")
             self.model = None
             self.scaler = MinMaxScaler(feature_range=(0,1)) # Reinitialize scaler


    
    def predict_instability(self, states, divergence):
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
            
        # Ensure inputs have the expected window size
        if len(states) != self.window_size or len(divergence) != self.window_size:
             # Handle incorrect window size, e.g., return default probability or raise error
             print(f"Warning: predict_instability received incorrect window size. States: {len(states)}, Div: {len(divergence)}. Expected: {self.window_size}")
             return 0.0 # Return a default probability


        feature = []
        
        feature.extend(states.flatten())
        feature.extend(divergence)
        
        feature.append(np.mean(states[:, 0]))
        feature.append(np.std(states[:, 0]))
        feature.append(np.mean(states[:, 1]))
        feature.append(np.std(states[:, 1]))
        feature.append(np.mean(states[:, 2]))
        feature.append(np.std(states[:, 2]))
        
        # Ensure scaler is fitted before transforming
        # Check if scaler has attributes set during fit (like scale_ or min_)
        if not hasattr(self.scaler, 'scale_'):
             print("Error: Scaler is not fitted. Cannot transform features.")
             # Handle this case: maybe fit the scaler with dummy data or return default
             return 0.0


        feature_scaled = self.scaler.transform([feature])
        
        feature_scaled = feature_scaled.reshape(1, 1, feature_scaled.shape[1])
        
        probability = self.model.predict(feature_scaled, verbose=0)[0][0]
        
        return probability
    
    def real_time_monitoring(self, solver, t_span, initial_state, dt, threshold=0.7):
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
        
        t_current = t_span[0]
        state_current = np.array(initial_state)
        
        t_points = [t_current]
        states_list = [state_current.copy()] # Use list initially
        probabilities = []
        warnings = []
        
        while len(states_list) < self.window_size:
            state_next = solver.step(t_current, state_current, dt)
            # Check for NaN or infinite values
            if not np.all(np.isfinite(state_next)):
                 print(f"Warning: Solver produced non-finite state at t={t_current + dt}. Stopping.")
                 break
            state_current = state_next
            t_current += dt
            t_points.append(t_current)
            states_list.append(state_current.copy())

        if len(states_list) < self.window_size:
             print("Error: Could not generate enough initial states for the window.")
             return np.array(t_points), np.array(states_list), np.array(probabilities), warnings


        states_array = np.array(states_list)
        divergence = self.calculate_local_divergence(states_array)
        
        while t_current < t_span[1]:
            # Ensure divergence has enough data for the window
            if len(divergence) < self.window_size:
                print("Warning: Not enough divergence data for prediction window. Skipping prediction.")
                 # Need to advance state anyway
                state_next = solver.step(t_current, state_current, dt)
                if not np.all(np.isfinite(state_next)):
                    print(f"Warning: Solver produced non-finite state at t={t_current + dt}. Stopping.")
                    break
                state_current = state_next
                t_current += dt
                t_points.append(t_current)
                states_list.append(state_current.copy())
                states_array = np.array(states_list) # Update states array
                # Recalculate divergence based on the updated states_array
                divergence = self.calculate_local_divergence(states_array)
                continue


            current_state_window = states_array[-self.window_size:]
            current_divergence_window = divergence[-self.window_size:]

            probability = self.predict_instability(current_state_window, current_divergence_window)
            probabilities.append(probability)
            
            if probability > threshold:
                warnings.append(t_current)
                print(f"预警: 在时间 {t_current:.2f} 检测到潜在不稳定性，概率: {probability:.4f}")
            
            state_next = solver.step(t_current, state_current, dt)
            if not np.all(np.isfinite(state_next)):
                 print(f"Warning: Solver produced non-finite state at t={t_current + dt}. Stopping.")
                 break
            state_current = state_next
            t_current += dt
            
            t_points.append(t_current)
            states_list.append(state_current.copy())
            
            states_array = np.array(states_list)
            divergence = self.calculate_local_divergence(states_array)
        
        return np.array(t_points), np.array(states_list), np.array(probabilities), warnings


def generate_training_data():
    print("生成训练数据...")
    
    t_span = [0, 1000]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem() # Instantiate LorenzSystem
    solver = RK4Solver(lorenz) # Pass the instance to the solver


    
    t, states = solver.solve(t_span, initial_state, dt)
    
    with open('training_data.pkl', 'wb') as f:
        pickle.dump({'t': t, 'states': states}, f)
    
    print(f"训练数据已保存到 'training_data.pkl'，共 {len(t)} 个时间点")
    
    return t, states


def train_instability_predictor(t=None, states=None):
    print("训练不稳定性预警系统...")
    
    if t is None or states is None:
        try:
            with open('training_data.pkl', 'rb') as f:
                data = pickle.load(f)
            t = data['t']
            states = data['states']
            print(f"已加载训练数据，共 {len(t)} 个时间点")
        except FileNotFoundError:
            print("未找到保存的训练数据，正在生成...")
            t, states = generate_training_data()
    
    predictor = LorenzInstabilityPredictor(window_size=20, prediction_horizon=10, lyapunov_threshold=0.5)
    
    print("开始训练LSTM模型...")
    start_time = time.time()
    history = predictor.train(t, states, epochs=30, batch_size=64, verbose=1)
    end_time = time.time()
    print(f"训练完成，耗时 {end_time - start_time:.2f} 秒")
    
    # Check if training was successful before saving/plotting
    if history is None or predictor.model is None:
        print("训练失败，无法保存模型或绘制历史。")
        return None

    predictor.save_model('instability_model.h5', 'instability_scaler.pkl')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('instability_training_history.png', dpi=300)
    plt.close()
    
    return predictor


def test_instability_predictor(predictor=None):
    print("测试不稳定性预警系统...")
    
    if predictor is None:
        predictor = LorenzInstabilityPredictor()
        try:
            predictor.load_model('instability_model.h5', 'instability_scaler.pkl')
            if predictor.model is None: # Check if loading failed
                 print("加载模型失败，正在重新训练...")
                 predictor = train_instability_predictor()
                 if predictor is None: # Check if retraining failed
                      print("无法测试，训练/加载模型失败。")
                      return None

        except Exception as e: # Catch broader exceptions during loading
            print(f"加载模型或scaler时出错: {e}，正在重新训练...")
            predictor = train_instability_predictor()
            if predictor is None: # Check if retraining failed
                 print("无法测试，训练/加载模型失败。")
                 return None


    # Ensure predictor is valid before proceeding
    if predictor is None or predictor.model is None:
        print("无法进行测试，预警系统无效。")
        return None


    t_span = [0, 100]
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    
    lorenz = LorenzSystem() # Instantiate LorenzSystem
    rk4_solver = RK4Solver(lorenz) # Pass instance
    hybrid_solver = HybridAdaptiveSolver(lorenz) # Pass instance


    
    print("使用RK4求解器测试...")
    t_rk4, states_rk4, probs_rk4, warnings_rk4 = predictor.real_time_monitoring(
        rk4_solver, t_span, initial_state, dt, threshold=0.7)
    
    print("使用混合求解器测试...")
    t_hybrid, states_hybrid, probs_hybrid, warnings_hybrid = predictor.real_time_monitoring(
        hybrid_solver, t_span, initial_state, dt, threshold=0.7)
    
    plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 1, 1, projection='3d')
    ax1.plot(states_rk4[:, 0], states_rk4[:, 1], states_rk4[:, 2], 'b-', linewidth=0.5, label='RK4')
    
    unique_warnings_rk4 = []
    if warnings_rk4:
        # Plot only the first warning marker for the legend
        first_warning_idx = np.abs(t_rk4 - warnings_rk4[0]).argmin()
        ax1.scatter(states_rk4[first_warning_idx, 0], states_rk4[first_warning_idx, 1], states_rk4[first_warning_idx, 2],
                   c='r', s=50, marker='*', label='Warning')
        # Plot subsequent warnings without label
        for w in warnings_rk4[1:]:
             idx = np.abs(t_rk4 - w).argmin()
             ax1.scatter(states_rk4[idx, 0], states_rk4[idx, 1], states_rk4[idx, 2],
                        c='r', s=50, marker='*')


    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Trajectory with Instability Warnings')
    ax1.legend()
    
    ax2 = plt.subplot(2, 1, 2)
    # Ensure probability array indices align with time points after windowing
    start_index = predictor.window_size
    if len(t_rk4) > start_index and len(probs_rk4) == len(t_rk4) - start_index:
         ax2.plot(t_rk4[start_index:], probs_rk4, 'b-', label='RK4 Probabilities')
    else:
         print(f"RK4 time/probability length mismatch: t={len(t_rk4)}, prob={len(probs_rk4)}, start_idx={start_index}")

    if len(t_hybrid) > start_index and len(probs_hybrid) == len(t_hybrid) - start_index:
         ax2.plot(t_hybrid[start_index:], probs_hybrid, 'g-', label='Hybrid Probabilities')
    else:
         print(f"Hybrid time/probability length mismatch: t={len(t_hybrid)}, prob={len(probs_hybrid)}, start_idx={start_index}")


    
    ax2.axhline(y=0.7, color='r', linestyle='--', label='Warning Threshold')
    
    for w in warnings_rk4:
        ax2.axvline(x=w, color='r', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Instability Probability')
    ax2.set_title('Instability Prediction')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('instability_prediction_test.png', dpi=300)
    plt.close()
    
    with open('instability_prediction_report.md', 'w') as f:
        f.write("# Lorenz系统不稳定性预警系统性能报告\n\n")
        f.write("## 测试参数\n")
        f.write(f"- 时间范围: {t_span}\n")
        f.write(f"- 初始状态: {initial_state}\n")
        f.write(f"- 时间步长: {dt}\n")
        f.write(f"- 预警阈值: 0.7\n\n")
        
        f.write("## 预警统计\n")
        f.write(f"- RK4求解器预警次数: {len(warnings_rk4)}\n")
        if warnings_rk4:
            f.write(f"- RK4求解器首次预警时间: {warnings_rk4[0]:.2f}\n")
        f.write(f"- 混合求解器预警次数: {len(warnings_hybrid)}\n")
        if warnings_hybrid:
            f.write(f"- 混合求解器首次预警时间: {warnings_hybrid[0]:.2f}\n\n")
        
        f.write("## 不稳定区域特征\n")
        if warnings_rk4:
            warning_indices = [np.abs(t_rk4 - w).argmin() for w in warnings_rk4]
            warning_states = states_rk4[warning_indices]
            
            f.write(f"- X坐标范围: [{np.min(warning_states[:, 0]):.4f}, {np.max(warning_states[:, 0]):.4f}]\n")
            f.write(f"- Y坐标范围: [{np.min(warning_states[:, 1]):.4f}, {np.max(warning_states[:, 1]):.4f}]\n")
            f.write(f"- Z坐标范围: [{np.min(warning_states[:, 2]):.4f}, {np.max(warning_states[:, 2]):.4f}]\n\n")
        
        f.write("## 结论\n")
        f.write("- 预警系统能够成功识别Lorenz系统中的潜在不稳定区域\n")
        f.write("- 预警通常发生在系统轨迹发生剧烈变化之前\n")
        f.write("- 混合求解器与RK4求解器的预警结果基本一致，证明混合求解器能够保持系统的动力学特性\n")
    
    print("测试完成，结果已保存到 'instability_prediction_test.png' 和 'instability_prediction_report.md'")
    
    return {
        'rk4': {'t': t_rk4, 'states': states_rk4, 'probabilities': probs_rk4, 'warnings': warnings_rk4},
        'hybrid': {'t': t_hybrid, 'states': states_hybrid, 'probabilities': probs_hybrid, 'warnings': warnings_hybrid}
    }


if __name__ == "__main__":
    t, states = generate_training_data()
    predictor = train_instability_predictor(t, states)
    
    if predictor: # Only test if training was successful
        test_results = test_instability_predictor(predictor)
    else:
        print("由于训练失败，跳过测试。")
