# 求解器稳定性和Lyapunov指数分析

## 评估参数
- Lyapunov指数计算时间范围: [0, 50]
- 物理合理性评估时间范围: [0, 100]
- 初始状态: [1.0, 1.0, 1.0]
- 基准时间步长: 0.01

## 稳定性和Lyapunov指数对比表格

| Solver | Max Lyapunov | Critical dt | Energy Variation | Z Always Positive | Instability |
| --- | --- | --- | --- | --- | --- |
| Energy Preserving | 4.933180 | 0.500000 | 0.074359 | 是 | 否 |
| RK4 | 10.536704 | 0.100000 | 0.762368 | 是 | 否 |
| RK2 | 10.796218 | 0.050000 | 0.762309 | 是 | 否 |
| Euler | 11.263659 | 0.020000 | 0.786380 | 是 | 否 |
| Symplectic | 11.279039 | 0.050000 | 0.793916 | 是 | 否 |
| Midpoint | 11.393356 | 0.050000 | 0.765945 | 是 | 否 |

## 说明
- Max Lyapunov: 最大局部Lyapunov指数，值越大表示系统越敏感
- Critical dt: 临界时间步长，超过此值求解器会变得不稳定
- Energy Variation: 伪能量的变异系数（标准差/平均值），越小越稳定
- Z Always Positive: Lorenz系统中z坐标是否始终为正
- Instability: 是否出现数值不稳定性（NaN或无穷大）
