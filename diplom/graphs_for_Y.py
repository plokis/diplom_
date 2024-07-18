import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 80000)

m25 = np.loadtxt('./data/2_nagr_q1_01_M_phi90_Rh100_delta10_sh_sh.txt', dtype='f4')
m50 = np.loadtxt('./data/2_nagr_q2_01_M_phi90_Rh100_delta10_sh_sh.txt', dtype='f4')
m100 = np.loadtxt('./data/2_nagr_q3_01_M_phi90_Rh100_delta10_sh_sh.txt', dtype='f4')
# m150 = np.loadtxt('./data/2_nagr_q3_01_M_phi90_Rh150_delta10_sh_sh.txt', dtype='f4')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
plt.plot(x, m25, label='R / h = 100, q = 1')
plt.plot(x, m50, label='R / h = 100, q = 2')
plt.plot(x, m100, label='R / h = 100, q = 3')
# plt.plot(x, m150, label='R / h = 150, q = 3')
ax.set_title("Эпюры перемещений ur при разных значений нагрузок для длины оболочки равной 90 градусов")
ax.grid(True)
plt.legend(loc='best')
plt.show()

