# Lorenz系统求解器增强功能使用说明

本文档提供了Lorenz系统求解器增强功能的使用说明，包括可视化增强、误差分布可视化和参数空间探索三个主要模块。

## 1. 项目结构

增强功能包含以下三个主要Python文件：

- `visualization_enhanced.py`: 提供动态可视化、3D交互式可视化和轨迹动画功能
- `error_visualization.py`: 提供误差热图、稳定性区域可视化和误差分析功能
- `parameter_space_explorer.py`: 提供参数敏感性分析、临界参数分析和稳定性研究功能

## 2. 可视化增强功能

### 2.1 基本用法

```python
# 导入可视化模块
from visualization_enhanced import LorenzVisualizer, generate_visualization_data, create_basic_visualizations, create_animations

# 生成可视化数据
results = generate_visualization_data()

# 创建基本可视化
create_basic_visualizations(results)

# 创建动画可视化
create_animations(results)
```

### 2.2 主要功能

- **3D轨迹可视化**：绘制Lorenz系统的三维轨迹，可使用颜色表示时间
- **相位图**：绘制XY、XZ、YZ平面的相位图和时间序列
- **吸引子结构可视化**：绘制吸引子结构，包括3D视图和密度图
- **轨迹动画**：创建轨迹的动态演化动画
- **多视角动画**：同时展示3D视图和三个平面投影的动画
- **旋转视角动画**：创建围绕吸引子旋转的动画

### 2.3 自定义可视化

```python
# 创建可视化器
visualizer = LorenzVisualizer(output_dir="my_visualizations")

# 自定义3D轨迹可视化
visualizer.plot_3d_trajectory(t, states, 
                             title="My Custom Trajectory", 
                             filename="custom_trajectory.png",
                             show_time_color=True,
                             azimuth=-45, elevation=30)

# 自定义轨迹动画
visualizer.create_trajectory_animation(t, states,
                                     title="My Custom Animation",
                                     filename="custom_animation.gif",
                                     fps=30, dpi=120)
```

## 3. 误差分布可视化功能

### 3.1 基本用法

```python
# 导入误差可视化模块
from error_visualization import ErrorVisualizer, generate_error_visualization_data, create_error_visualizations

# 生成误差可视化数据
results = generate_error_visualization_data()

# 创建误差可视化
create_error_visualizations(results)
```

### 3.2 主要功能

- **误差分布图**：比较不同求解器的平均误差和最大误差
- **误差热图**：在不同平面上可视化误差分布
- **3D误差可视化**：在3D空间中可视化误差分布
- **稳定性区域**：分析不同步长下各求解器的稳定性
- **参数空间稳定性热图**：分析参数空间中的稳定性区域
- **误差收敛性分析**：分析不同步长下误差的收敛行为

### 3.3 自定义误差分析

```python
# 创建误差可视化器
visualizer = ErrorVisualizer(output_dir="my_error_analysis")

# 自定义稳定性区域分析
dt_values = np.logspace(-4, -1, 30)  # 从0.0001到0.1的30个步长值
solver_names = ["Euler", "RK2", "RK4", "Energy"]

stability_results = visualizer.plot_stability_regions(dt_values, solver_names,
                                                   title="Custom Stability Analysis",
                                                   filename="custom_stability.png")

# 自定义误差收敛性分析
error_convergence = visualizer.plot_error_convergence(dt_values, solver_names,
                                                   title="Custom Error Convergence",
                                                   filename="custom_convergence.png")
```

## 4. 参数空间探索功能

### 4.1 基本用法

```python
# 导入参数空间探索模块
from parameter_space_explorer import ParameterSpaceExplorer, explore_parameter_space, create_parameter_space_report

# 探索参数空间
results = explore_parameter_space()

# 创建参数空间探索报告
create_parameter_space_report(results)
```

### 4.2 主要功能

- **参数敏感性分析**：研究sigma、rho和beta参数对求解器性能的影响
- **临界参数分析**：探索系统在临界参数值附近的行为变化
- **二维参数空间分析**：分析sigma-rho参数空间中的系统行为
- **数值稳定性分析**：研究不同步长和参数值下的数值稳定性
- **长期行为分析**：分析不同求解器在长时间积分下的性能

### 4.3 自定义参数探索

```python
# 创建参数空间探索器
explorer = ParameterSpaceExplorer(output_dir="my_parameter_analysis")

# 自定义参数敏感性分析
sigma_values = np.linspace(5, 25, 20)
solver_names = ["Euler", "RK2", "RK4", "Energy", "Structure"]

sigma_results = explorer.explore_parameter_sensitivity(
    'sigma', sigma_values, solver_names,
    title="Custom Sigma Sensitivity Analysis",
    filename="custom_sigma_sensitivity.png"
)

# 自定义临界参数分析
rho_critical_values = np.linspace(0.1, 50, 500)

critical_results = explorer.explore_critical_parameters(
    rho_critical_values, solver_name="RK4",
    title="Custom Bifurcation Analysis",
    filename="custom_bifurcation.png"
)
```

## 5. 运行完整分析

要运行所有增强功能的完整分析，可以按以下顺序执行Python文件：

```bash
# 运行可视化增强
python visualization_enhanced.py

# 运行误差分布可视化
python error_visualization.py

# 运行参数空间探索
python parameter_space_explorer.py
```

注意：完整分析可能需要较长时间，特别是参数空间探索部分。如果只需要特定功能，可以导入相应模块并调用特定函数。

## 6. 输出文件

所有可视化结果将保存在以下目录：

- 可视化增强：`visualizations/`
- 误差分布可视化：`error_visualizations/`
- 参数空间探索：`parameter_space/`

主要报告文件：
- `parameter_space_report.md`：参数空间探索的详细报告

数据文件：
- `visualization_data.pkl`：可视化数据
- `error_visualization_data.pkl`：误差可视化数据
- `parameter_space_results.pkl`：参数空间探索结果

## 7. 自定义与扩展

所有模块都设计为可扩展的，您可以：

1. 添加新的可视化方法到`LorenzVisualizer`类
2. 扩展`ErrorVisualizer`类以包含更多误差分析方法
3. 在`ParameterSpaceExplorer`类中添加新的参数探索功能

例如，添加新的可视化方法：

```python
def plot_custom_visualization(self, t, states, ...):
    # 自定义可视化代码
    ...
    
# 将方法添加到LorenzVisualizer类
LorenzVisualizer.plot_custom_visualization = plot_custom_visualization
```

## 8. 性能考虑

- 生成高质量动画可能需要较长时间和较大内存
- 参数空间探索在分析大量参数组合时计算密集
- 对于大规模分析，考虑减少参数点数量或使用并行计算

## 9. 依赖库

这些增强功能依赖以下Python库：

- numpy
- matplotlib
- scipy
- pickle (Python标准库)
- time (Python标准库)
- os (Python标准库)

确保这些库已正确安装以获得最佳体验。
