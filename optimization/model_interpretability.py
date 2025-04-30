import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import joblib
import pickle
import time
import inspect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_interpretability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelInterpretability")

def apply_inspect_patch():
    if not hasattr(inspect, 'getargspec'):
        def getargspec_shim(func):
            sig = inspect.signature(func)
            params = sig.parameters
            args = []
            defaults = []
            for param_name, param in params.items():
                args.append(param_name)
                if param.default != inspect.Parameter.empty:
                    defaults.append(param.default)
            return inspect.ArgSpec(args=args, varargs=None, keywords=None, defaults=tuple(defaults) or None)

        inspect.getargspec = getargspec_shim
        logger.info("Applied global inspect patch for getargspec compatibility with Python 3.12")

apply_inspect_patch()

import eli5
import eli5.sklearn
from eli5.sklearn import PermutationImportance

try:
    from pdpbox import pdp, info_plots
    logger.info("成功导入pdpbox")
except ImportError:
    logger.warning("无法导入pdpbox，部分功能将不可用")
    class PDPModule:
        def __init__(self):
            pass

        def pdp_isolate(self, *args, **kwargs):
            logger.error("pdpbox未安装，pdp_isolate方法不可用")
            return None

        def pdp_plot(self, *args, **kwargs):
            logger.error("pdpbox未安装，pdp_plot方法不可用")
            return None, None

        def pdp_interact(self, *args, **kwargs):
            logger.error("pdpbox未安装，pdp_interact方法不可用")
            return None

        def pdp_interact_plot(self, *args, **kwargs):
            logger.error("pdpbox未安装，pdp_interact_plot方法不可用")
            return None, None

    pdp = PDPModule()
    info_plots = None

try:
    from IPython.display import display, clear_output, HTML
    logger.info("成功导入IPython显示功能")
except ImportError:
    logger.warning("无法导入IPython显示功能，将使用替代方案")
    class HTML:
        def __init__(self, html_string):
            self.html_string = html_string

        def _repr_html_(self):
            return self.html_string

        def __str__(self):
            return self.html_string

    def display(obj):
        print(obj)

    def clear_output(wait=False):
        pass

try:
    from lorenz_system import LorenzSystem
    from numerical_solvers import RK4Solver as RungeKutta4
    from numerical_solvers import HybridSolver as AdaptiveRungeKutta
    logger.info("成功导入Lorenz系统模块")
except ImportError as e:
    logger.warning(f"无法导入Lorenz系统模块: {e}")
    class LorenzSystem:
        def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
            self.sigma = sigma
            self.rho = rho
            self.beta = beta

    class RungeKutta4:
        def __init__(self, system=None):
            self.system = system

        def solve(self, t_span, initial_state, dt):
            return [], []

    class AdaptiveRungeKutta:
        def __init__(self, system=None):
            self.system = system

        def solve(self, t_span, initial_state, dt):
            return [], []

class FeatureImportanceAnalyzer:
    def __init__(self, output_dir="model_interpretability_plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.importance_results = {}
        self.feature_names = None

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def analyze_permutation_importance(self, model, X, y, n_repeats=10, random_state=42):
        logger.info("开始分析排列重要性...")
        start_time = time.time()
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state
        )
        elapsed_time = time.time() - start_time
        logger.info(f"排列重要性分析完成，耗时: {elapsed_time:.2f}秒")
        self.importance_results['permutation'] = perm_importance
        self._plot_permutation_importance(perm_importance)
        return perm_importance

    def analyze_shap_importance(self, model, X, model_type='tree', n_samples=100):
        logger.info(f"开始分析SHAP重要性 (模型类型: {model_type})...")
        try:
            start_time = time.time()
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            elif model_type == 'deep':
                sample_size = min(n_samples, len(X))
                background = X[:sample_size]
                explainer = shap.DeepExplainer(model, background)
                test_size = min(n_samples*2, len(X))
                shap_values = explainer.shap_values(X[:test_size])
            else:
                sample_size = min(n_samples, len(X))
                explainer = shap.KernelExplainer(model.predict_proba, X[:sample_size])
                test_size = min(n_samples*2, len(X))
                shap_values = explainer.shap_values(X[:test_size])
            elapsed_time = time.time() - start_time
            logger.info(f"SHAP重要性分析完成，耗时: {elapsed_time:.2f}秒")
            self.importance_results['shap'] = {
                'explainer': explainer,
                'shap_values': shap_values
            }
            self._plot_shap_summary(explainer, shap_values, X)
            return shap_values
        except Exception as e:
            logger.error(f"SHAP重要性分析失败: {e}")
            return {}

    def analyze_eli5_importance(self, model, X, y, feature_names=None):
        logger.info("开始分析ELI5特征重要性...")
        start_time = time.time()
        perm = PermutationImportance(model, random_state=42).fit(X, y)
        elapsed_time = time.time() - start_time
        logger.info(f"ELI5特征重要性分析完成，耗时: {elapsed_time:.2f}秒")
        self.importance_results['eli5'] = perm
        feature_names = feature_names or self.feature_names
        weights = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=feature_names))
        weights.to_csv(os.path.join(self.output_dir, "eli5_weights.csv"))
        self._plot_eli5_importance(weights)
        return perm

    def analyze_pdp(self, model, X, feature_indices=None, feature_names=None, n_samples=100):
        logger.info("开始分析部分依赖图...")
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        pdp_results = {}
        start_time = time.time()
        for i in feature_indices:
            feature_name = feature_names[i]
            logger.info(f"计算特征 '{feature_name}' 的部分依赖图...")
            pdp_result = pdp.pdp_isolate(
                model=model,
                dataset=X_df,
                model_features=feature_names,
                feature=feature_name,
                num_grid_points=20,
                n_jobs=-1,
                sample_size=min(n_samples, len(X))
            )
            pdp_results[feature_name] = pdp_result
            fig, axes = pdp.pdp_plot(
                pdp_result, feature_name, plot_lines=True,
                frac_to_plot=0.5, plot_pts_dist=True
            )
            plt.savefig(os.path.join(self.output_dir, f"pdp_{feature_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        elapsed_time = time.time() - start_time
        logger.info(f"部分依赖图分析完成，耗时: {elapsed_time:.2f}秒")
        self.importance_results['pdp'] = pdp_results
        return pdp_results

    def analyze_ice(self, model, X, feature_indices=None, feature_names=None, n_samples=100):
        logger.info("开始分析个体条件期望图...")
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        ice_results = {}
        start_time = time.time()
        for i in feature_indices:
            feature_name = feature_names[i]
            logger.info(f"计算特征 '{feature_name}' 的ICE图...")
            ice_result = pdp.pdp_isolate(
                model=model,
                dataset=X_df,
                model_features=feature_names,
                feature=feature_name,
                num_grid_points=20,
                n_jobs=-1,
                sample_size=min(n_samples, len(X))
            )
            ice_results[feature_name] = ice_result
            fig, axes = pdp.pdp_plot(
                ice_result, feature_name, plot_lines=True,
                frac_to_plot=0.5, plot_pts_dist=True,
                center=True, plot_params={'alpha': 0.3, 'linewidth': 0.5}
            )
            plt.savefig(os.path.join(self.output_dir, f"ice_{feature_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        elapsed_time = time.time() - start_time
        logger.info(f"个体条件期望图分析完成，耗时: {elapsed_time:.2f}秒")
        self.importance_results['ice'] = ice_results
        return ice_results

    def analyze_feature_interactions(self, model, X, feature_pairs=None, feature_names=None, n_samples=100):
        logger.info("开始分析特征交互...")
        if feature_pairs is None:
            n_features = min(5, X.shape[1])
            feature_pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        interaction_results = {}
        start_time = time.time()
        for i, j in feature_pairs:
            feature1 = feature_names[i]
            feature2 = feature_names[j]
            logger.info(f"计算特征 '{feature1}' 和 '{feature2}' 的交互...")
            inter_result = pdp.pdp_interact(
                model=model,
                dataset=X_df,
                model_features=feature_names,
                features=[feature1, feature2],
                num_grid_points=[10, 10],
                n_jobs=-1,
                sample_size=min(n_samples, len(X))
            )
            interaction_results[(feature1, feature2)] = inter_result
            fig, axes = pdp.pdp_interact_plot(
                inter_result, [feature1, feature2],
                plot_type='contour', x_quantile=True,
                plot_pdp=True
            )
            plt.savefig(
                os.path.join(self.output_dir, f"interaction_{feature1.replace(' ', '_')}_{feature2.replace(' ', '_')}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        elapsed_time = time.time() - start_time
        logger.info(f"特征交互分析完成，耗时: {elapsed_time:.2f}秒")
        self.importance_results['interaction'] = interaction_results
        return interaction_results

    def _plot_permutation_importance(self, perm_importance):
        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(perm_importance.importances_mean))]
        indices = perm_importance.importances_mean.argsort()[::-1]
        plt.figure(figsize=(12, 8))
        plt.barh(
            range(len(indices)),
            perm_importance.importances_mean[indices],
            xerr=perm_importance.importances_std[indices],
            align='center'
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title('排列重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "permutation_importance.png"), dpi=300)
        plt.close()
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in range(len(perm_importance.importances_mean))],
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        importance_df.to_csv(os.path.join(self.output_dir, "permutation_importance.csv"), index=False)

    def _plot_shap_summary(self, explainer, shap_values, X, plot_type='bar'):
        try:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            feature_names = self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
            plt.figure(figsize=(12, 8))
            if plot_type == 'bar':
                if isinstance(shap_values, list):
                    values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    values_to_plot = shap_values
                if len(values_to_plot.shape) > 2:
                    values_to_plot = values_to_plot[0]
                if len(values_to_plot.shape) == 1:
                    values_to_plot = values_to_plot.reshape(1, -1)
                if values_to_plot.shape[1] != X_df.shape[1]:
                    if values_to_plot.shape[1] > X_df.shape[1]:
                        values_to_plot = values_to_plot[:, :X_df.shape[1]]
                    else:
                        padding = np.zeros((values_to_plot.shape[0], X_df.shape[1] - values_to_plot.shape[1]))
                        values_to_plot = np.hstack([values_to_plot, padding])
                shap.plots.bar(shap.Explanation(values=values_to_plot,
                                               data=X_df.values,
                                               feature_names=feature_names),
                              show=False)
            else:
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X_df, show=False)
                else:
                    shap_explanation = shap.Explanation(values=shap_values,
                                                       data=X_df,
                                                       feature_names=feature_names)
                    shap.plots.beeswarm(shap_explanation, show=False)
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f"shap_summary_{plot_type}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            if isinstance(shap_values, list):
                values_to_save = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values_to_save = shap_values
            shap_df = pd.DataFrame(values_to_save, columns=feature_names)
            shap_df.to_csv(os.path.join(self.output_dir, "shap_values.csv"), index=False)
            return plot_path
        except Exception as e:
            logger.warning(f"SHAP摘要图绘制失败: {e}")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"SHAP摘要图绘制失败: {e}",
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            error_path = os.path.join(self.output_dir, "shap_summary_error.png")
            plt.savefig(error_path, dpi=300)
            plt.close()
            return error_path

    def _plot_eli5_importance(self, weights_df):
        plt.figure(figsize=(12, 8))
        weights_df = weights_df.sort_values(by='weight', ascending=False)
        plt.barh(
            range(len(weights_df)),
            weights_df['weight'],
            align='center'
        )
        plt.yticks(range(len(weights_df)), weights_df['feature'])
        plt.xlabel('权重')
        plt.title('ELI5特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "eli5_importance.png"), dpi=300)
        plt.close()

    def get_top_features(self, method='permutation', n_top=5):
        if method not in self.importance_results:
            logger.warning(f"方法 '{method}' 的结果不可用")
            return []
        feature_names = self.feature_names
        if feature_names is None:
            if method == 'permutation':
                n_features = len(self.importance_results[method].importances_mean)
            elif method == 'shap':
                shap_values = self.importance_results[method]['shap_values']
                n_features = shap_values[0].shape[1] if isinstance(shap_values, list) else shap_values.shape[1]
            elif method == 'eli5':
                n_features = len(self.importance_results[method].feature_importances_)
            feature_names = [f"Feature {i}" for i in range(n_features)]
        if method == 'permutation':
            indices = self.importance_results[method].importances_mean.argsort()[::-1]
            return [feature_names[i] for i in indices[:n_top]]
        elif method == 'shap':
            try:
                shap_values = self.importance_results[method]['shap_values']
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[0]
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                indices = mean_abs_shap.argsort()[::-1]
                valid_indices = [i for i in indices if i < len(feature_names)]
                return [feature_names[i] for i in valid_indices[:n_top]]
            except Exception as e:
                logger.error(f"计算SHAP特征重要性时出错: {e}")
                return feature_names[:min(n_top, len(feature_names))]
        elif method == 'eli5':
            perm = self.importance_results[method]
            weights = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=feature_names))
            weights = weights.sort_values(by='weight', ascending=False)
            return list(weights['feature'][:n_top])
        return []

    def get_importance_report(self):
        report = "特征重要性分析报告:\n\n"
        if 'permutation' in self.importance_results:
            report += "排列重要性:\n"
            feature_names = self.feature_names or [f"Feature {i}" for i in range(len(self.importance_results['permutation'].importances_mean))]
            indices = self.importance_results['permutation'].importances_mean.argsort()[::-1]
            for i in indices:
                report += f"  {feature_names[i]}: {self.importance_results['permutation'].importances_mean[i]:.6f} ± {self.importance_results['permutation'].importances_std[i]:.6f}\n"
            report += "\n"
        if 'shap' in self.importance_results:
            report += "SHAP重要性:\n"
            feature_names = self.feature_names
            if feature_names is None:
                shap_values = self.importance_results['shap']['shap_values']
                n_features = shap_values[0].shape[1] if isinstance(shap_values, list) else shap_values.shape[1]
                feature_names = [f"Feature {i}" for i in range(n_features)]
            try:
                shap_values = self.importance_results['shap']['shap_values']
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[0]
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
                if shap_values.shape[1] != len(feature_names):
                    if shap_values.shape[1] > len(feature_names):
                        shap_values = shap_values[:, :len(feature_names)]
                    else:
                        padding = np.zeros((shap_values.shape[0], len(feature_names) - shap_values.shape[1]))
                        shap_values = np.hstack([shap_values, padding])
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                indices = mean_abs_shap.argsort()[::-1]
                valid_indices = [i for i in indices if i < len(feature_names)]
                for i in valid_indices:
                    report += f"  {feature_names[i]}: {mean_abs_shap[i]:.6f}\n"
            except Exception as e:
                logger.error(f"计算SHAP特征重要性时出错: {e}")
                report += f"  无法计算SHAP重要性: {e}\n"
            report += "\n"
        if 'eli5' in self.importance_results:
            report += "ELI5重要性:\n"
            feature_names = self.feature_names or [f"Feature {i}" for i in range(len(self.importance_results['eli5'].feature_importances_))]
            perm = self.importance_results['eli5']
            weights = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names=feature_names))
            weights = weights.sort_values(by='weight', ascending=False)
            for _, row in weights.iterrows():
                report += f"  {row['feature']}: {row['weight']:.6f}\n"
            report += "\n"
        report += "综合分析:\n"
        top_features = {}
        for method in ['permutation', 'shap', 'eli5']:
            if method in self.importance_results:
                top_features[method] = self.get_top_features(method, n_top=5)
        if top_features:
            all_top_features = [f for features in top_features.values() for f in features]
            feature_counts = {f: all_top_features.count(f) for f in set(all_top_features)}
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            report += "  在多个方法中都排名靠前的特征:\n"
            for feature, count in sorted_features:
                if count > 1:
                    report += f"    {feature}: 在{count}种方法中排名靠前\n"
            report += "\n  各方法的前5个特征:\n"
            for method, features in top_features.items():
                report += f"    {method}: {', '.join(str(f) for f in features)}\n"
        return report

    def save_results(self, results=None, output_dir=None):
        results_to_save = results if results is not None else self.importance_results
        output_dir = output_dir or self.output_dir
        filename = os.path.join(output_dir, "feature_importance_results.pkl")
        os.makedirs(output_dir, exist_ok=True)
        serializable_results = {}
        if 'permutation' in results_to_save:
            serializable_results['permutation'] = {
                'importances_mean': results_to_save['permutation'].importances_mean,
                'importances_std': results_to_save['permutation'].importances_std
            }
        if 'shap' in results_to_save:
            serializable_results['shap'] = {
                'shap_values': results_to_save['shap']['shap_values']
            }
        if 'eli5' in results_to_save:
            serializable_results['eli5'] = results_to_save['eli5']
        if 'pdp' in results_to_save:
            serializable_results['pdp'] = results_to_save['pdp']
        if 'ice' in results_to_save:
            serializable_results['ice'] = results_to_save['ice']
        if 'interaction' in results_to_save:
            serializable_results['interaction'] = results_to_save['interaction']
        try:
            with open(filename, 'wb') as f:
                pickle.dump(serializable_results, f)
            logger.info(f"分析结果已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存结果时出错: {e}")

class ModelExplainer:
    def __init__(self, output_dir="model_interpretability_plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.explainers = {}
        self.feature_names = None

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def create_lime_explainer(self, X_train, feature_names=None, class_names=None, mode='classification'):
        logger.info("创建LIME解释器...")
        try:
            feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X_train.shape[1])]
            if mode == 'classification':
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode=mode
                )
            else:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    mode=mode
                )
            self.explainers['lime'] = explainer
            return explainer
        except Exception as e:
            logger.error(f"创建LIME解释器时出错: {e}")
            return None

    def create_shap_explainer(self, model, X_train, model_type='tree'):
        logger.info(f"创建SHAP解释器 (模型类型: {model_type})...")
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_train)
        elif model_type == 'deep':
            background = X_train[:min(100, len(X_train))]
            explainer = shap.DeepExplainer(model, background)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_train[:min(100, len(X_train))])
        self.explainers['shap'] = explainer
        return explainer

    def explain_instance_lime(self, instance, model, num_features=10, predict_fn=None):
        logger.info("使用LIME解释实例...")
        if 'lime' not in self.explainers:
            logger.error("LIME解释器尚未创建")
            return None
        predict_fn = predict_fn or (model.predict_proba if hasattr(model, 'predict_proba') else model.predict)
        explanation = self.explainers['lime'].explain_instance(
            instance, predict_fn, num_features=num_features
        )
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lime_explanation.png"), dpi=300, bbox_inches='tight')
        plt.close()
        return explanation

    def explain_instance_shap(self, instance, plot_type='waterfall'):
        logger.info(f"使用SHAP解释实例 (图表类型: {plot_type})...")
        try:
            if 'shap' not in self.explainers:
                logger.error("SHAP解释器尚未创建")
                return {}
            explainer = self.explainers['shap']
            if isinstance(instance, list):
                instance_array = np.array([instance]) if not isinstance(instance[0], (list, np.ndarray)) else np.array(instance)
                instance_data = instance[0] if isinstance(instance[0], (list, np.ndarray)) else instance
            elif isinstance(instance, np.ndarray):
                instance_array = instance if len(instance.shape) > 1 else instance.reshape(1, -1)
                instance_data = instance[0] if len(instance.shape) > 1 else instance
            else:
                instance_array = np.array([instance])
                instance_data = instance
            shap_values = explainer.shap_values(instance_array)
            feature_names = self.feature_names or [f"Feature {i}" for i in range(len(instance_data))]
            plt.figure(figsize=(12, 6))
            if plot_type == 'waterfall':
                if isinstance(shap_values, list):
                    shap.plots.waterfall(shap.Explanation(values=shap_values[1] if len(shap_values) > 1 else shap_values[0],
                                                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value,
                                                         data=instance_data,
                                                         feature_names=feature_names), show=False)
                else:
                    shap.plots.waterfall(shap.Explanation(values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                                                         base_values=explainer.expected_value,
                                                         data=instance_data,
                                                         feature_names=feature_names), show=False)
            elif plot_type == 'force':
                if isinstance(shap_values, list):
                    shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value,
                                   shap_values[1] if len(shap_values) > 1 else shap_values[0],
                                   instance_data, feature_names=feature_names, matplotlib=True, show=False)
                else:
                    shap.force_plot(explainer.expected_value,
                                   shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                                   instance_data, feature_names=feature_names, matplotlib=True, show=False)
            elif plot_type == 'decision':
                if isinstance(shap_values, list):
                    shap.decision_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1 else explainer.expected_value,
                                      shap_values[1] if len(shap_values) > 1 else shap_values[0],
                                      feature_names=feature_names, show=False)
                else:
                    shap.decision_plot(explainer.expected_value,
                                      shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                                      feature_names=feature_names, show=False)
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f"shap_{plot_type}_explanation.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return shap_values
        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            return {}

    def explain_model_performance(self, model, X, y, feature_names=None):
        logger.info("解释模型性能...")
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            y_pred = (y_proba[:, 1] > 0.5).astype(int) if y_proba.shape[1] > 1 else (y_proba > 0.5).astype(int)
            y_score = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
        else:
            y_pred = model.predict(X)
            y_score = y_pred
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_score)
        }
        cm = confusion_matrix(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('接收者操作特征曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"), dpi=300)
        plt.close()
        precision, recall, _ = precision_recall_curve(y, y_score)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR曲线 (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "pr_curve.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, "performance_metrics.csv"), index=False)
        return metrics

    def visualize_decision_boundary(self, model, X, y, feature_indices=None, resolution=100):
        logger.info("可视化决策边界...")
        if feature_indices is None:
            feature_indices = [0, 1]
        feature_names = self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        X_vis = X[:, feature_indices]
        x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
        y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        if X.shape[1] > 2:
            X_mean = np.mean(X, axis=0)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            X_grid = np.tile(X_mean, (grid_points.shape[0], 1))
            X_grid[:, feature_indices] = grid_points
        else:
            X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(X_grid)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_grid)
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu_r)
        plt.colorbar(contour)
        scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu_r)
        plt.xlabel(feature_names[feature_indices[0]])
        plt.ylabel(feature_names[feature_indices[1]])
        plt.title('决策边界')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "decision_boundary.png"), dpi=300)
        plt.close()
        return contour

    def visualize_decision_surface_3d(self, model, X, y, feature_indices=None, resolution=20):
        logger.info("可视化3D决策表面...")
        if feature_indices is None:
            feature_indices = [0, 1, 2]
        feature_names = self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        X_vis = X[:, feature_indices]
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=y, cmap=plt.cm.RdBu_r, s=50, alpha=0.6)
        x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
        y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1
        z_min, z_max = X_vis[:, 2].min() - 0.1, X_vis[:, 2].max() + 0.1
        if hasattr(model, 'predict_proba'):
            grid_points = np.array([[x, y, z] for x in np.linspace(x_min, x_max, resolution)
                                   for y in np.linspace(y_min, y_max, resolution)
                                   for z in np.linspace(z_min, z_max, resolution)])
            if X.shape[1] > 3:
                X_mean = np.mean(X, axis=0)
                X_grid = np.tile(X_mean, (grid_points.shape[0], 1))
                X_grid[:, feature_indices] = grid_points
            else:
                X_grid = grid_points
            probs = model.predict_proba(X_grid)[:, 1]
            threshold_points = grid_points[np.abs(probs - 0.5) < 0.05]
            if len(threshold_points) > 0:
                ax.scatter(threshold_points[:, 0], threshold_points[:, 1], threshold_points[:, 2],
                          c='green', alpha=0.3, s=10)
        ax.set_xlabel(feature_names[feature_indices[0]])
        ax.set_ylabel(feature_names[feature_indices[1]])
        ax.set_zlabel(feature_names[feature_indices[2]])
        ax.set_title('3D决策表面')
        plt.colorbar(scatter, ax=ax, label='类别')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "decision_surface_3d.png"), dpi=300)
        plt.close()
        return fig

    def get_explanation_report(self, instance_explanations=None):
        report = "模型解释报告:\n\n"
        if os.path.exists(os.path.join(self.output_dir, "performance_metrics.csv")):
            metrics_df = pd.read_csv(os.path.join(self.output_dir, "performance_metrics.csv"))
            report += "性能指标:\n"
            for column in metrics_df.columns:
                report += f"  {column}: {metrics_df[column].values[0]:.4f}\n"
            report += "\n"
        if instance_explanations and 'lime' in instance_explanations:
            lime_exp = instance_explanations['lime']
            report += "LIME解释:\n"
            for feature, weight in lime_exp.as_list():
                report += f"  {feature}: {weight:.4f}\n"
            report += "\n"
        if instance_explanations and 'shap' in instance_explanations:
            shap_values = instance_explanations['shap']
            report += "SHAP解释:\n"
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            feature_names = self.feature_names or [f"Feature {i}" for i in range(len(shap_values[0]))]
            for i, feature in enumerate(feature_names):
                report += f"  {feature}: {shap_values[0][i]:.4f}\n"
            report += "\n"
        return report

class LorenzModelInterpreter:
    def __init__(self, output_dir="lorenz_model_interpretability"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.importance_analyzer = FeatureImportanceAnalyzer(
            output_dir=os.path.join(output_dir, "feature_importance")
        )
        self.model_explainer = ModelExplainer(
            output_dir=os.path.join(output_dir, "model_explanation")
        )
        self.feature_names = None
        self.feature_descriptions = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def set_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_model(self, model):
        self.model = model

    def set_feature_info(self, feature_names, feature_descriptions=None):
        self.feature_names = feature_names
        self.feature_descriptions = feature_descriptions
        self.importance_analyzer.set_feature_names(feature_names)
        self.model_explainer.set_feature_names(feature_names)

    def analyze_feature_importance(self, methods=None):
        logger.info("分析Lorenz系统模型的特征重要性...")
        if self.model is None or self.X_test is None or self.y_test is None:
            logger.error("模型或数据尚未设置")
            return {}
        if methods is None:
            methods = ['permutation', 'shap', 'eli5', 'pdp', 'ice', 'interaction']
        results = {}
        if 'permutation' in methods:
            perm_importance = self.importance_analyzer.analyze_permutation_importance(
                self.model, self.X_test, self.y_test
            )
            results['permutation'] = perm_importance
        if 'shap' in methods:
            model_type = 'tree' if hasattr(self.model, 'estimators_') else 'linear' if hasattr(self.model, 'coef_') else 'deep' if hasattr(self.model, 'layers') else 'other'
            shap_values = self.importance_analyzer.analyze_shap_importance(
                self.model, self.X_test, model_type=model_type
            )
            results['shap'] = shap_values
        if 'eli5' in methods:
            eli5_importance = self.importance_analyzer.analyze_eli5_importance(
                self.model, self.X_test, self.y_test, self.feature_names
            )
            results['eli5'] = eli5_importance
        if 'pdp' in methods:
            top_features = list(range(min(5, self.X_test.shape[1])))
            pdp_results = self.importance_analyzer.analyze_pdp(
                self.model, self.X_test, feature_indices=top_features, feature_names=self.feature_names
            )
            results['pdp'] = pdp_results
        if 'ice' in methods:
            top_features = list(range(min(5, self.X_test.shape[1])))
            ice_results = self.importance_analyzer.analyze_ice(
                self.model, self.X_test, feature_indices=top_features, feature_names=self.feature_names
            )
            results['ice'] = ice_results
        if 'interaction' in methods:
            top_features = list(range(min(3, self.X_test.shape[1])))
            feature_pairs = [(i, j) for i in top_features for j in top_features if i < j]
            interaction_results = self.importance_analyzer.analyze_feature_interactions(
                self.model, self.X_test, feature_pairs=feature_pairs, feature_names=self.feature_names
            )
            results['interaction'] = interaction_results
        self.importance_analyzer.save_results(results)
        report = self.importance_analyzer.get_importance_report()
        with open(os.path.join(self.output_dir, "feature_importance_report.txt"), "w") as f:
            f.write(report)
        return results

    def explain_model(self, instance=None, visualize=True):
        logger.info("解释Lorenz系统模型...")
        if self.model is None or self.X_train is None or self.X_test is None or self.y_test is None:
            logger.error("模型或数据尚未设置")
            return {}
        results = {}
        model_type = 'tree' if hasattr(self.model, 'estimators_') else 'linear' if hasattr(self.model, 'coef_') else 'deep' if hasattr(self.model, 'layers') else 'other'
        self.model_explainer.create_shap_explainer(self.model, self.X_train, model_type=model_type)
        if instance is not None:
            lime_explainer = self.model_explainer.create_lime_explainer(
                self.X_train, feature_names=self.feature_names, class_names=['Stable', 'Unstable']
            )
            if lime_explainer:
                lime_exp = self.model_explainer.explain_instance_lime(instance, self.model)
                results['lime'] = lime_exp
            shap_values = self.model_explainer.explain_instance_shap(instance)
            results['shap'] = shap_values
        performance_metrics = self.model_explainer.explain_model_performance(
            self.model, self.X_test, self.y_test, feature_names=self.feature_names
        )
        results['performance'] = performance_metrics
        if visualize:
            self.model_explainer.visualize_decision_boundary(self.model, self.X_test, self.y_test)
            if self.X_test.shape[1] >= 3:
                self.model_explainer.visualize_decision_surface_3d(self.model, self.X_test, self.y_test)
        instance_explanations = {k: v for k, v in results.items() if k in ['lime', 'shap']}
        report = self.model_explainer.get_explanation_report(instance_explanations)
        with open(os.path.join(self.output_dir, "instance_explanation_report.txt"), "w") as f:
            f.write(report)
        return results

    def analyze_time_horizon_effects(self, time_horizons=None, n_samples=100):
        logger.info("分析时间范围对模型的影响...")
        if self.model is None or self.X_train is None or self.y_train is None:
            logger.error("模型或数据尚未设置")
            return {}
        if time_horizons is None:
            time_horizons = [1, 5, 10, 20]
        results = {}
        performance_over_time = {}
        importance_over_time = {}
        X_train_sample = self.X_train[:min(n_samples, len(self.X_train))]
        y_train_sample = self.y_train[:min(n_samples, len(self.y_train))]
        for horizon in time_horizons:
            logger.info(f"分析时间范围: {horizon}")
            X_horizon = self._generate_time_horizon_data(X_train_sample, horizon)
            if X_horizon.shape[0] == 0:
                logger.warning(f"时间范围 {horizon} 的数据生成失败，跳过")
                continue
            perm_importance = self.importance_analyzer.analyze_permutation_importance(
                self.model, X_horizon, y_train_sample
            )
            importance_over_time[horizon] = perm_importance.importances_mean
            y_pred = self.model.predict(X_horizon)
            performance_over_time[horizon] = {
                'accuracy': accuracy_score(y_train_sample, y_pred),
                'precision': precision_score(y_train_sample, y_pred, zero_division=0),
                'recall': recall_score(y_train_sample, y_pred, zero_division=0),
                'f1': f1_score(y_train_sample, y_pred, zero_division=0)
            }
        results['performance'] = performance_over_time
        results['importance'] = importance_over_time
        self._plot_time_horizon_effects(results)
        report = self._get_time_horizon_report(results)
        with open(os.path.join(self.output_dir, "time_horizon_report.txt"), "w") as f:
            f.write(report)
        return results

    def analyze_parameter_sensitivity(self, param_ranges=None, n_points=10):
        logger.info("分析Lorenz系统参数敏感性...")
        if self.model is None or self.X_train is None or self.y_train is None:
            logger.error("模型或数据尚未设置")
            return {}
        if param_ranges is None:
            param_ranges = {
                'sigma': (5.0, 15.0),
                'rho': (20.0, 35.0),
                'beta': (2.0, 3.0)
            }
        results = {}
        for param, (min_val, max_val) in param_ranges.items():
            logger.info(f"分析参数: {param}")
            param_values = np.linspace(min_val, max_val, n_points)
            performance = []
            for val in param_values:
                X_sensitivity = self._generate_sensitivity_data(param, val)
                if X_sensitivity.shape[0] == 0:
                    logger.warning(f"参数 {param} 值 {val} 的数据生成失败，跳过")
                    continue
                y_pred = self.model.predict(X_sensitivity)
                perf = {
                    'accuracy': accuracy_score(self.y_train[:len(y_pred)], y_pred),
                    'f1': f1_score(self.y_train[:len(y_pred)], y_pred, zero_division=0)
                }
                performance.append(perf)
            results[param] = {
                'values': param_values,
                'performance': performance
            }
            self._plot_parameter_sensitivity(param, param_values, performance)
        report = self._get_parameter_sensitivity_report(results)
        with open(os.path.join(self.output_dir, "parameter_sensitivity_report.txt"), "w") as f:
            f.write(report)
        return results

    def generate_comprehensive_report(self):
        logger.info("生成综合报告...")
        report = "Lorenz系统模型综合解释报告\n\n"
        report += "1. 特征重要性分析\n"
        importance_report_path = os.path.join(self.output_dir, "feature_importance_report.txt")
        if os.path.exists(importance_report_path):
            with open(importance_report_path, "r") as f:
                report += f.read()
        else:
            report += "  未找到特征重要性报告\n"
        report += "\n2. 模型解释\n"
        explanation_report_path = os.path.join(self.output_dir, "instance_explanation_report.txt")
        if os.path.exists(explanation_report_path):
            with open(explanation_report_path, "r") as f:
                report += f.read()
        else:
            report += "  未找到实例解释报告\n"
        report += "\n3. 时间范围效应\n"
        time_horizon_report_path = os.path.join(self.output_dir, "time_horizon_report.txt")
        if os.path.exists(time_horizon_report_path):
            with open(time_horizon_report_path, "r") as f:
                report += f.read()
        else:
            report += "  未找到时间范围效应报告\n"
        report += "\n4. 参数敏感性分析\n"
        sensitivity_report_path = os.path.join(self.output_dir, "parameter_sensitivity_report.txt")
        if os.path.exists(sensitivity_report_path):
            with open(sensitivity_report_path, "r") as f:
                report += f.read()
        else:
            report += "  未找到参数敏感性报告\n"
        report += "\n5. 总结与建议\n"
        if self.feature_names and self.importance_analyzer.importance_results:
            top_features = self.importance_analyzer.get_top_features('permutation')
            report += f"  最重要的特征: {', '.join(top_features)}\n"
            if self.feature_descriptions:
                report += "  特征描述:\n"
                for feature in top_features:
                    desc = self.feature_descriptions.get(feature, "无描述")
                    report += f"    {feature}: {desc}\n"
            report += "  建议: 关注这些特征以优化模型性能和稳定性预测。\n"
        else:
            report += "  无法生成总结，缺少特征重要性数据。\n"
        with open(os.path.join(self.output_dir, "comprehensive_report.txt"), "w") as f:
            f.write(report)
        return report

    def _generate_time_horizon_data(self, X, horizon):
        try:
            X_horizon = np.copy(X)
            if X_horizon.shape[0] > horizon:
                X_horizon = X_horizon[:horizon]
            return X_horizon
        except Exception as e:
            logger.error(f"生成时间范围数据失败: {e}")
            return np.array([])

    def _generate_sensitivity_data(self, param, value):
        try:
            lorenz = LorenzSystem(
                sigma=value if param == 'sigma' else 10.0,
                rho=value if param == 'rho' else 28.0,
                beta=value if param == 'beta' else 8.0/3.0
            )
            solver = RungeKutta4(lorenz)
            t_span = [0, 10]
            initial_state = [1.0, 1.0, 1.0]
            dt = 0.01
            t, states = solver.solve(t_span, initial_state, dt)
            X_sensitivity = np.array(states)
            return X_sensitivity
        except Exception as e:
            logger.error(f"生成参数敏感性数据失败: {e}")
            return np.array([])

    def _plot_time_horizon_effects(self, results):
        plt.figure(figsize=(12, 8))
        horizons = list(results['performance'].keys())
        accuracies = [results['performance'][h]['accuracy'] for h in horizons]
        plt.plot(horizons, accuracies, marker='o', label='准确率')
        plt.xlabel('时间范围')
        plt.ylabel('性能')
        plt.title('时间范围对模型性能的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "time_horizon_performance.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(self.feature_names or [f"Feature {i}" for i in range(len(results['importance'][horizons[0]]))]):
            importance = [results['importance'][h][i] for h in horizons]
            plt.plot(horizons, importance, marker='o', label=feature)
        plt.xlabel('时间范围')
        plt.ylabel('特征重要性')
        plt.title('时间范围对特征重要性的影响')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "time_horizon_importance.png"), dpi=300)
        plt.close()

    def _plot_parameter_sensitivity(self, param, values, performance):
        plt.figure(figsize=(10, 8))
        accuracies = [p['accuracy'] for p in performance]
        plt.plot(values, accuracies, marker='o', label='准确率')
        plt.xlabel(f'参数 {param}')
        plt.ylabel('性能')
        plt.title(f'参数 {param} 对模型性能的影响')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"param_impact_{param}.png"), dpi=300)
        plt.close()

    def _get_time_horizon_report(self, results):
        report = "时间范围效应分析报告:\n\n"
        report += "性能随时间范围变化:\n"
        for horizon, metrics in results['performance'].items():
            report += f"  时间范围 {horizon}:\n"
            for metric, value in metrics.items():
                report += f"    {metric}: {value:.4f}\n"
        report += "\n特征重要性随时间范围变化:\n"
        feature_names = self.feature_names or [f"Feature {i}" for i in range(len(next(iter(results['importance'].values()))))]
        for i, feature in enumerate(feature_names):
            report += f"  {feature}:\n"
            for horizon in results['importance']:
                report += f"    时间范围 {horizon}: {results['importance'][horizon][i]:.4f}\n"
        return report

    def _get_parameter_sensitivity_report(self, results):
        report = "参数敏感性分析报告:\n\n"
        for param, data in results.items():
            report += f"参数 {param}:\n"
            best_value = None
            best_accuracy = -1
            for val, perf in zip(data['values'], data['performance']):
                report += f"  值 {val:.2f}: 准确率 = {perf['accuracy']:.4f}, F1 = {perf['f1']:.4f}\n"
                if perf['accuracy'] > best_accuracy:
                    best_accuracy = perf['accuracy']
                    best_value = val
            report += f"  最佳值: {best_value:.2f} (准确率 = {best_accuracy:.4f})\n\n"
        return report

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    interpreter = LorenzModelInterpreter()
    interpreter.set_data(X_train, y_train, X_test, y_test)
    interpreter.set_model(model)
    interpreter.set_feature_info(
        feature_names=['x', 'y', 'z'],
        feature_descriptions={
            'x': 'X坐标',
            'y': 'Y坐标',
            'z': 'Z坐标'
        }
    )

    importance_results = interpreter.analyze_feature_importance()
    instance = X_test[0]
    explanation_results = interpreter.explain_model(instance=instance)
    time_results = interpreter.analyze_time_horizon_effects()
    param_results = interpreter.analyze_parameter_sensitivity()
    report = interpreter.generate_comprehensive_report()
    print(report)
