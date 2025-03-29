# 用 iterated inverse interpolation 解 x = e^(-x)

# 已知資料點 (x, e^{-x}) → 轉換為 (y = x - e^{-x}, x)
data = [
    (-0.440818, 0.3),
    (-0.270320, 0.4),
    (-0.106531, 0.5),
    ( 0.051188, 0.6)
]

# 拆成兩組
y_vals = [row[0] for row in data]  # y = x - e^{-x}
x_vals = [row[1] for row in data]  # 對應的 x 值

# 差商表（牛頓插值法）
def newton_divided_diff(x_vals, y_vals):
    n = len(x_vals)
    coef = list(y_vals)  # 複製
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_vals[i] - x_vals[i-j])
    return coef

# 插值函數求值
def newton_evaluate(x, x_vals, coef):
    n = len(coef)
    result = coef[-1]
    for i in range(n-2, -1, -1):
        result = result * (x - x_vals[i]) + coef[i]
    return result

# 執行反插值：我們已知 y = 0，要求對應的 x
coeffs = newton_divided_diff(y_vals, x_vals)
x_approx = newton_evaluate(0, y_vals, coeffs)

print(f"使用 iterated inverse interpolation 解 x = e^(-x)")
print(f"當 y = 0 時，近似解為 x ≈ {x_approx:.6f}")
