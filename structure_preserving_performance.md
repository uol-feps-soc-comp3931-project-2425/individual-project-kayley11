# 结构保持积分器性能对比

## 评估参数
- 时间范围: [0, 100]
- 初始状态: [1.0, 1.0, 1.0]
- 时间步长: 0.01

## 体积收缩率对比

| Solver | Volume Contraction Rate | Error from Theoretical |
| --- | --- | --- |
| Theoretical | -13.666667 | 0.000000 |
| RK4 | 0.250791 | 13.917457 |
| Structure Preserving | nan | nan |
| Enhanced Structure Preserving | nan | nan |

## 伪能量稳定性对比

| Solver | Energy Variation Coefficient |
| --- | --- |
| RK4 | 0.762368 |
| Structure Preserving | nan |
| Enhanced Structure Preserving | nan |

## 结构保持积分器修正统计

| Correction Type | Structure Preserving | Enhanced Structure Preserving |
| --- | --- | --- |
| Volume Correction | 100.00% | 100.00% |
| Constraint Enforcement | 0.00% | 0.00% |
| Attractor Preservation | 0.08% | 0.11% |
