from math import factorial

# 已知資料點與目標
x_vals = [0.698, 0.733, 0.768, 0.803]
f_vals = [0.7661, 0.7432, 0.7193, 0.6946]
x_target = 0.750
true_val = 0.7317

# 通用 Lagrange 插值函數
def lagrange_interp_general(x, x_vals, f_vals):
    result = 0
    n = len(x_vals)
    for i in range(n):
        term = f_vals[i]
        for j in range(n):
            if j != i:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        result += term
    return result

# 誤差界估算函數（假設最高導數 ≤ 1）
def lagrange_error_bound(x, x_vals, max_deriv=1):
    product = 1
    for xi in x_vals:
        product *= (x - xi)
    return abs(max_deriv * product / factorial(len(x_vals)))

# 逐次計算 degree 1 到 3 的結果
for degree in range(1, 4):
    x_sub = x_vals[:degree+1]
    f_sub = f_vals[:degree+1]
    approx = lagrange_interp_general(x_target, x_sub, f_sub)
    error = abs(approx - true_val)
    bound = lagrange_error_bound(x_target, x_sub)

    print(f"\n--- Degree {degree} ---")
    print(f"近似值 P_{degree}({x_target}) = {approx:.6f}")
    print(f"真實誤差 = |{true_val} - {approx:.6f}| = {error:.6f}")
    print(f"誤差界（估計） = {bound:.6e}")
