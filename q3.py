import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar, root_scalar
import matplotlib.pyplot as plt

# 題目給定資料
T = np.array([0, 3, 5, 8, 13])            # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])     # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])        # 速度 (ft/s)

# 建立 Hermite 插值器
hermite_poly = CubicHermiteSpline(T, D, V)

# === (a) 預測 t = 10 時的位置與速度 ===
t_eval = 10
pos_at_10 = hermite_poly(t_eval)                    # 位置
speed_at_10 = hermite_poly.derivative()(t_eval)     # 速度（導數）

print("=== (a) 預測 t = 10 秒時 ===")
print(f"位置：{pos_at_10:.2f} ft")
print(f"速度：{speed_at_10:.2f} ft/s")

# === (b) 是否超過 55 mi/h（約 80.67 ft/s）===
speed_limit = 80.67

# 定義速度是否超速的函數
def exceeds_speed(t):
    return hermite_poly.derivative()(t) - speed_limit

# 嘗試用 root_scalar 找第一次超過的時間
try:
    result = root_scalar(exceeds_speed, bracket=[3, 5], method='brentq')
    exceed_time = result.root
    print("\n=== (b) 速度是否超過 55 mi/h？===")
    print("✅ 有超過")
    print(f"第一次超過時間：{exceed_time:.2f} 秒")
except ValueError:
    print("\n=== (b) 速度是否超過 55 mi/h？===")
    print("❌ 沒有超過")

# === (c) 預測最大速度 ===
result = minimize_scalar(lambda t: -hermite_poly.derivative()(t), bounds=(T[0], T[-1]), method='bounded')
max_speed = -result.fun
max_time = result.x

print("\n=== (c) 預測最大速度 ===")
print(f"最大速度：{max_speed:.2f} ft/s")
print(f"出現時間：{max_time:.2f} 秒")

# === 額外繪圖 ===
t_vals = np.linspace(0, 13, 300)
speeds = hermite_poly.derivative()(t_vals)

