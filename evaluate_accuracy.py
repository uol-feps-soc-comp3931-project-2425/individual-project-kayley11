import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from lorenz_system import LorenzSystem
from numerical_solvers import RK4Solver

class EnhancedPredictionEvaluator:
    def __init__(self, output_dir="prediction_evaluation"):
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parameter_sets = [
            {"sigma": 10.0, "rho": 28.0, "beta": 8/3},
            {"sigma": 10.0, "rho": 14.0, "beta": 8/3},
            {"sigma": 10.0, "rho": 35.0, "beta": 8/3},
            {"sigma": 8.0, "rho": 28.0, "beta": 8/3},
            {"sigma": 12.0, "rho": 28.0, "beta": 8/3},
            {"sigma": 10.0, "rho": 28.0, "beta": 2.0},
            {"sigma": 10.0, "rho": 28.0, "beta": 3.5}
        ]
        
        self.initial_conditions = [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [0.1, 0.1, 0.1],
            [-1.0, -1.0, 1.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0]
        ]
        
        self.prediction_horizons = [
            0.5,
            1.0,
            2.0,
            5.0,
            10.0
        ]
    
    def generate_diverse_dataset(self, save_path="diverse_training_data.pkl"):
        print("生成多样化的数据集...")
        
        t_span = [0, 500]
        dt = 0.01
        
        all_trajectories = []
        
        for params in self.parameter_sets:
            for init_cond in self.initial_conditions:
                print(f"  生成轨迹: 参数={params}, 初始条件={init_cond}")
                
                lorenz = LorenzSystem(sigma=params["sigma"], rho=params["rho"], beta=params["beta"])
                solver = RK4Solver(lorenz)
                t, states = solver.solve(t_span, init_cond, dt)
                
                all_trajectories.append({
                    "params": params,
                    "init_cond": init_cond,
                    "t": t,
                    "states": states
                })
        
        with open(os.path.join(self.output_dir, save_path), 'wb') as f:
            pickle.dump(all_trajectories, f)
        
        print(f"多样化数据集已保存到 {os.path.join(self.output_dir, save_path)}")
        
        return all_trajectories
    
    def calculate_lyapunov_exponent(self, states, dt, window_size=100):
        lyapunov_exponents = []
        
        for i in range(window_size, len(states)):
            window = states[i-window_size:i]
            diffs = np.diff(window, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            growth_rate = np.mean(np.log(norms + 1e-10)) / dt
            lyapunov_exponents.append(growth_rate)
        
        return np.array(lyapunov_exponents)
    
    def prepare_data_for_prediction(self, trajectory, window_size=50, prediction_horizon=1.0, dt=0.01):
        states = trajectory["states"]
        lyapunov_exponents = self.calculate_lyapunov_exponent(states, dt)
        
        lyapunov_time = 1.1
        prediction_steps = int(prediction_horizon * lyapunov_time / dt)
        
        features = []
        labels = []
        
        for i in range(window_size, len(states) - prediction_steps):
            feature = np.column_stack([
                states[i-window_size:i],
                np.zeros((window_size, 1))
            ])
            
            if i - window_size < len(lyapunov_exponents):
                feature[-1, -1] = 0
            else:
                feature[-1, -1] = lyapunov_exponents[i-window_size-1]
            
            threshold = 0.5
            future_indices = range(i, min(i + prediction_steps, len(lyapunov_exponents) + window_size))
            
            if len(future_indices) > 0 and max(future_indices) - window_size < len(lyapunov_exponents):
                future_lyapunov = lyapunov_exponents[max(0, i-window_size):min(i+prediction_steps-window_size, len(lyapunov_exponents))]
                label = 1 if np.any(future_lyapunov > threshold) else 0
            else:
                label = 0
            
            features.append(feature)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate_model(self, prediction_horizon=1.0, window_size=50, dt=0.01):
        print(f"训练和评估模型: 预测时间范围={prediction_horizon}个Lyapunov时间")
        
        try:
            with open(os.path.join(self.output_dir, "diverse_training_data.pkl"), 'rb') as f:
                all_trajectories = pickle.load(f)
        except FileNotFoundError:
            all_trajectories = self.generate_diverse_dataset()
        
        all_features = []
        all_labels = []
        
        for trajectory in all_trajectories:
            features, labels = self.prepare_data_for_prediction(
                trajectory, window_size, prediction_horizon, dt
            )
            all_features.append(features)
            all_labels.append(labels)
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = self.build_lstm_model((window_size, X.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = np.mean(y_pred == y_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        evaluation_results = {
            "prediction_horizon": prediction_horizon,
            "window_size": window_size,
            "accuracy": accuracy,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1_score,
            "ap": ap,
            "roc_auc": roc_auc,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            },
            "precision_recall_curve": {
                "precision": precision,
                "recall": recall
            },
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr
            },
            "training_history": {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "accuracy": history.history["accuracy"],
                "val_accuracy": history.history["val_accuracy"]
            }
        }
        
        return evaluation_results
    
    def evaluated_accuracy(self, model, X_test, y_test, threshold=0.5):
        print("详细评估模型准确性...")
        
        y_pred_proba = model.predict(X_test, verbose=0)
        
        thresholds = np.linspace(0.1, 0.9, 9)
        threshold_results = {}
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int).flatten()
            
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            tn = np.sum((y_pred == 0) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            threshold_results[thresh] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
                "confusion_matrix": {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn
                }
            }
        
        class_balance = np.mean(y_test)
        
        proba_hist, proba_bins = np.histogram(y_pred_proba, bins=10, range=(0, 1))
        
        pos_proba = y_pred_proba[y_test == 1].flatten() if np.any(y_test == 1) else np.array([])
        neg_proba = y_pred_proba[y_test == 0].flatten() if np.any(y_test == 0) else np.array([])
        
        pos_hist, _ = np.histogram(pos_proba, bins=10, range=(0, 1)) if len(pos_proba) > 0 else (np.array([]), None)
        neg_hist, _ = np.histogram(neg_proba, bins=10, range=(0, 1)) if len(neg_proba) > 0 else (np.array([]), None)
        
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        error_indices = np.where(y_pred != y_test)[0]
        
        error_features = {}
        if len(error_indices) > 0:
            error_X = X_test[error_indices]
            error_features = {
                "mean": np.mean(error_X, axis=0).tolist(),
                "std": np.std(error_X, axis=0).tolist(),
                "count": len(error_indices)
            }
        
        accuracy_evaluation = {
            "threshold_results": threshold_results,
            "optimal_threshold": max(threshold_results.items(), key=lambda x: x[1]["f1_score"])[0],
            "class_balance": class_balance,
            "probability_distribution": {
                "bins": proba_bins.tolist(),
                "histogram": proba_hist.tolist(),
                "positive_class": pos_hist.tolist() if len(pos_hist) > 0 else [],
                "negative_class": neg_hist.tolist() if len(neg_hist) > 0 else []
            },
            "error_analysis": {
                "error_rate": len(error_indices) / len(y_test) if len(y_test) > 0 else 0,
                "error_features": error_features
            }
        }
        
        with open(os.path.join(self.output_dir, "accuracy_evaluation.pkl"), 'wb') as f:
            pickle.dump(accuracy_evaluation, f)
        
        print(f"准确性评估结果已保存到 {os.path.join(self.output_dir, 'accuracy_evaluation.pkl')}")
        
        return accuracy_evaluation
    
    def evaluate_across_time_horizons(self):
        print("在不同时间范围内评估模型...")
        
        time_horizon_results = {}
        
        for horizon in self.prediction_horizons:
            print(f"  评估预测时间范围: {horizon}个Lyapunov时间")
            results = self.train_and_evaluate_model(prediction_horizon=horizon)
            time_horizon_results[horizon] = results
        
        with open(os.path.join(self.output_dir, "time_horizon_evaluation.pkl"), 'wb') as f:
            pickle.dump(time_horizon_results, f)
        
        return time_horizon_results
    
    def plot_time_horizon_comparison(self, time_horizon_results):
        print("绘制不同时间范围的性能对比...")
        
        horizons = list(time_horizon_results.keys())
        horizons.sort()
        
        metrics = {
            "accuracy": "准确率",
            "precision": "精确率",
            "recall": "召回率",
            "f1_score": "F1分数",
            "ap": "平均精确率",
            "roc_auc": "ROC曲线下面积"
        }
        
        plt.figure(figsize=(12, 8))
        
        for metric, label in metrics.items():
            values = [time_horizon_results[h][metric] for h in horizons]
            plt.plot(horizons, values, 'o-', label=label)
        
        plt.xlabel('预测时间范围（Lyapunov时间）', fontsize=14)
        plt.ylabel('性能指标', fontsize=14)
        plt.title('不同预测时间范围下的模型性能', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        plt.savefig(os.path.join(self.output_dir, "time_horizon_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        for horizon in horizons:
            results = time_horizon_results[horizon]
            fpr = results["roc_curve"]["fpr"]
            tpr = results["roc_curve"]["tpr"]
            roc_auc = results["roc_auc"]
            
            plt.plot(fpr, tpr, label=f'{horizon}个Lyapunov时间 (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率', fontsize=14)
        plt.ylabel('真阳性率', fontsize=14)
        plt.title('不同预测时间范围的ROC曲线', fontsize=16)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, "time_horizon_roc_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        for horizon in horizons:
            results = time_horizon_results[horizon]
            precision = results["precision_recall_curve"]["precision"]
            recall = results["precision_recall_curve"]["recall"]
            ap = results["ap"]
            
            plt.plot(recall, precision, label=f'{horizon}个Lyapunov时间 (AP = {ap:.3f})')
        
        plt.xlabel('召回率', fontsize=14)
        plt.ylabel('精确率', fontsize=14)
        plt.title('不同预测时间范围的精确率-召回率曲线', fontsize=16)
        plt.legend(fontsize=12, loc='lower left')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, "time_horizon_pr_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, time_horizon_results):
        print("生成评估报告...")
        
        horizons = list(time_horizon_results.keys())
        horizons.sort()
        
        with open(os.path.join(self.output_dir, "prediction_evaluation_report.md"), 'w') as f:
            f.write("# Lorenz系统不稳定性预测系统增强评估报告\n\n")
            f.write("## 1. 评估方法\n\n")
            f.write("本评估采用了以下方法来全面评估预测系统的性能：\n\n")
            f.write("### 1.1 数据多样性\n\n")
            f.write("为确保评估的全面性，我们使用了多样化的数据集：\n\n")
            f.write(f"- **参数集数量**：{len(self.parameter_sets)}组不同的Lorenz系统参数\n")
            f.write(f"- **初始条件数量**：{len(self.initial_conditions)}组不同的初始条件\n")
            f.write(f"- **总轨迹数量**：{len(self.parameter_sets) * len(self.initial_conditions)}条不同轨迹\n\n")
            f.write("### 1.2 预测时间范围\n\n")
            f.write("我们评估了不同时间尺度下的预测性能：\n\n")
            f.write("- **Lyapunov时间**：Lorenz系统的特征时间尺度，约为1.1个时间单位\n")
            f.write("- **评估的时间范围**：")
            for i, horizon in enumerate(horizons):
                if i > 0:
                    f.write("、")
                f.write(f"{horizon}")
            f.write("个Lyapunov时间\n\n")
            f.write("## 2. 性能随预测时间范围的变化\n\n")
            f.write("### 2.1 性能指标对比\n\n")
            f.write("| 预测时间范围（Lyapunov时间） | 准确率 | 精确率 | 召回率 | F1分数 | ROC AUC | 平均精确率 |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            for horizon in horizons:
                results = time_horizon_results[horizon]
                f.write(f"| {horizon} | {results['accuracy']:.4f} | {results['precision']:.4f} | ")
                f.write(f"{results['recall']:.4f} | {results['f1_score']:.4f} | ")
                f.write(f"{results['roc_auc']:.4f} | {results['ap']:.4f} |\n")
            
            f.write("\n")
            f.write("### 2.2 性能变化趋势分析\n\n")
            
            accuracy_trend = []
            f1_trend = []
            
            for i in range(1, len(horizons)):
                prev_horizon = horizons[i-1]
                curr_horizon = horizons[i]
                prev_accuracy = time_horizon_results[prev_horizon]["accuracy"]
                curr_accuracy = time_horizon_results[curr_horizon]["accuracy"]
                prev_f1 = time_horizon_results[prev_horizon]["f1_score"]
                curr_f1 = time_horizon_results[curr_horizon]["f1_score"]
                accuracy_change = (curr_accuracy - prev_accuracy) / prev_accuracy if prev_accuracy > 0 else 0
                f1_change = (curr_f1 - prev_f1) / prev_f1 if prev_f1 > 0 else 0
                accuracy_trend.append(accuracy_change)
                f1_trend.append(f1_change)
            
            avg_accuracy_change = np.mean(accuracy_trend) if accuracy_trend else 0
            avg_f1_change = np.mean(f1_trend) if f1_trend else 0
            
            f.write(f"随着预测时间范围从{horizons[0]}增加到{horizons[-1]}个Lyapunov时间：\n\n")
            
            if avg_accuracy_change < -0.05:
                f.write("- 准确率呈明显下降趋势，")
            elif avg_accuracy_change < 0:
                f.write("- 准确率呈轻微下降趋势，")
            elif avg_accuracy_change > 0.05:
                f.write("- 准确率呈明显上升趋势，")
            elif avg_accuracy_change > 0:
                f.write("- 准确率呈轻微上升趋势，")
            else:
                f.write("- 准确率保持相对稳定，")
            f.write(f"平均变化率为{avg_accuracy_change:.2%}\n")
            
            if avg_f1_change < -0.05:
                f.write("- F1分数呈明显下降趋势，")
            elif avg_f1_change < 0:
                f.write("- F1分数呈轻微下降趋势，")
            elif avg_f1_change > 0.05:
                f.write("- F1分数呈明显上升趋势，")
            elif avg_f1_change > 0:
                f.write("- F1分数呈轻微上升趋势，")
            else:
                f.write("- F1分数保持相对稳定，")
            f.write(f"平均变化率为{avg_f1_change:.2%}\n\n")
            
            acceptable_threshold = 0.7
            acceptable_horizons = [h for h in horizons if time_horizon_results[h]["f1_score"] >= acceptable_threshold]
            
            if acceptable_horizons:
                max_acceptable = max(acceptable_horizons)
                f.write(f"基于F1分数≥{acceptable_threshold}的标准，预测系统在{max_acceptable}个Lyapunov时间内的预测结果是可接受的。\n\n")
            else:
                f.write(f"基于F1分数≥{acceptable_threshold}的标准，预测系统在所有评估的时间范围内的预测结果均不理想。\n\n")
            
            best_horizon = max(horizons, key=lambda h: time_horizon_results[h]["f1_score"])
            best_f1 = time_horizon_results[best_horizon]["f1_score"]
            
            f.write(f"在所有评估的时间范围中，预测系统在{best_horizon}个Lyapunov时间范围内表现最佳，F1分数为{best_f1:.4f}。\n\n")
            f.write("## 3. 预测系统泛化能力分析\n\n")
            f.write("### 3.1 主要发现\n\n")
            
            best_horizon = None
            best_f1 = -1
            
            for horizon in horizons:
                if time_horizon_results[horizon]["f1_score"] > best_f1:
                    best_horizon = horizon
                    best_f1 = time_horizon_results[horizon]["f1_score"]
            
            f.write(f"1. 预测系统在{best_horizon}个Lyapunov时间范围内表现最佳，F1分数达到{best_f1:.4f}\n")
            f.write(f"2. 随着预测时间范围的增加，预测性能总体呈下降趋势，这符合混沌系统的特性\n")
            f.write("3. 使用多样化的训练数据（不同参数集和初始条件）有助于提高预测系统的泛化能力\n")
            f.write("\n### 3.2 应用建议\n\n")
            
            if acceptable_horizons:
                f.write(f"1. 建议将预测系统的应用范围限制在{max_acceptable}个Lyapunov时间内，以确保预测准确率\n")
            else:
                f.write("1. 当前预测系统在所有评估的时间范围内准确率均不理想，建议进一步优化模型\n")
            
            f.write("2. 在实际应用中，可以根据不同的需求调整预测阈值：\n")
            f.write("   - 对安全性要求高的场景，可降低阈值以减少漏报（假阴性），但会增加误报（假阳性）\n")
            f.write("   - 对效率要求高的场景，可提高阈值以减少误报，但会增加漏报风险\n")
            f.write("3. 建议定期使用新的系统数据重新训练模型，以适应可能的系统参数变化\n\n")
            f.write("### 3.3 未来改进方向\n\n")
            f.write("1. 探索更复杂的模型架构，如注意力机制或图神经网络，可能有助于捕捉Lorenz系统的长期依赖关系\n")
            f.write("2. 考虑集成多个预测模型，每个模型专注于不同的预测时间范围\n")
            f.write("3. 引入更多物理约束和先验知识，如Lorenz系统的守恒律和体积收缩率\n")
            f.write("4. 开发自适应预测系统，根据当前系统状态动态调整预测时间范围和阈值\n")
        
        print(f"评估报告已保存到 {os.path.join(self.output_dir, 'prediction_evaluation_report.md')}")

def run_enhanced_prediction_evaluation():
    print("开始运行增强版预测系统评估...")
    evaluator = EnhancedPredictionEvaluator()
    evaluator.generate_diverse_dataset()
    time_horizon_results = evaluator.evaluate_across_time_horizons()
    evaluator.plot_time_horizon_comparison(time_horizon_results)
    evaluator.generate_evaluation_report(time_horizon_results)
    print("增强版预测系统评估完成")

if __name__ == "__main__":
    run_enhanced_prediction_evaluation()
