import math
import numpy as np

# 題目資料
all_x = np.array([0.698, 0.733, 0.768, 0.803])
all_y = np.array([0.7661, 0.7432, 0.7193, 0.6946])
x_target = 0.750

# Lagrange 插值函數
def lagrange_interp(x_vals, y_vals, x):
    n = len(x_vals)
    total = 0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        total += term
    return total

# 誤差上界估算
def lagrange_error_bound(x_vals, x_target, max_deriv_val):
    n = len(x_vals) - 1
    factorial = math.factorial(n + 1)
    product = 1
    for xi in x_vals:
        product *= abs(x_target - xi)
    return (max_deriv_val / factorial) * product

# 選出離目標點最近的 n+1 個節點
def select_nearest_points(all_x, all_y, x_target, degree):
    indices = np.argsort(np.abs(all_x - x_target))[:degree + 1]
    sorted_indices = np.sort(indices)  # 保持順序
    return all_x[sorted_indices], all_y[sorted_indices]

# 假設最大階導數值（最多 cos^(n)，最大值為 1）
max_derivs = [abs(math.cos(x_target))] * 4

# 主程式：degree 1 到 4
for deg in range(1, 5):
    x_used, y_used = select_nearest_points(all_x, all_y, x_target, deg)
    approx = lagrange_interp(x_used, y_used, x_target)
    err = lagrange_error_bound(x_used, x_target, max_derivs[deg - 1])
    
    print(f"\n=== Degree {deg} ===")
    print(f"選用節點 x: {x_used}")
    print(f"近似值 cos(0.750) ≈ {approx:.6f}")
    print(f"誤差上限 ≤ {err:.3e}")
