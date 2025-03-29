import numpy as np
from scipy.interpolate import KroghInterpolator
import matplotlib.pyplot as plt

# 資料：時間 T (s)、位置 D (ft)、速度 V (ft/s)
T = np.array([0, 3, 5, 8, 13])
D = np.array([0, 200, 375, 620, 990])
V = np.array([75, 77, 80, 74, 72])  # feet/second

# 建立 Hermite 插值資料：每個點對應兩個資料（位置與速度）
T_hermite = []
D_hermite = []
for t, d, v in zip(T, D, V):
    T_hermite.extend([t, t])      # 每點時間重複
    D_hermite.extend([d, v])      # 一個是函數值，一個是導數

T_hermite = np.array(T_hermite)
D_hermite = np.array(D_hermite)

# 建立 Hermite 插值器
hermite_poly = KroghInterpolator(T_hermite, D_hermite)

# (a) 預測 t = 10 秒的距離與速度
t_eval = 10
position_at_10 = hermite_poly(t_eval)
speed_at_10 = hermite_poly.derivatives(t_eval)[1]  # 第一導數是速度

print("=== (a) 預測 t = 10s 時的狀態 ===")
print(f"位置: {position_at_10:.2f} feet")
print(f"速度: {speed_at_10:.2f} ft/s")

# (b) 檢查是否超過 55 mi/h = 80.67 ft/s
t_fine = np.linspace(0, 13, 500)
speeds = np.array([hermite_poly.derivatives(t)[1] for t in t_fine])

exceeds_55mph = np.any(speeds > 80.67)
first_exceed_time = t_fine[speeds > 80.67][0] if exceeds_55mph else None

print("\n=== (b) 是否超過 55 mi/h? ===")
print(f"有超過: {exceeds_55mph}")
if exceeds_55mph:
    print(f"第一次超過的時間: {first_exceed_time:.2f} 秒")

# (c) 最大速度與時間
max_speed = np.max(speeds)
max_speed_time = t_fine[np.argmax(speeds)]

print("\n=== (c) 最大速度 ===")
print(f"最大速度: {max_speed:.2f} ft/s")
print(f"出現時間: {max_speed_time:.2f} 秒")

# 額外：畫出速度曲線
plt.plot(t_fine, speeds, label='Speed (ft/s)', color='blue')
plt.axhline(80.67, color='red', linestyle='--', label='55 mi/h = 80.67 ft/s')
plt.title("Speed Curve from Hermite Interpolation")
plt.xlabel("Time (s)")
plt.ylabel("Speed (ft/s)")
plt.legend()
plt.grid(True)
plt.show()
