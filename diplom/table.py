import numpy as np
from prettytable import PrettyTable
import xlsxwriter as xl

# Сравнение результатов для разных закреплений
# shsh - шарнир-шарнир, zz - заделка-заделка

MRh_shsh_o = []
MRh_zz_o = []
urRh_shsh_o = []
urRh_zz_o = []
uzRh_shsh_o = []
uzRh_zz_o = []

MRh_shsh_extr = []
MRh_zz_extr = []
urRh_shsh_extr = []
urRh_zz_extr = []
uzRh_shsh_extr = []
uzRh_zz_extr = []

for i in range(1, 5):
    MRh_shsh_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_M_phi90_Rh' + str(i) + '_delta10_sh_sh.txt', dtype='f4'))
    MRh_zz_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_M_phi90_Rh' + str(i) + '_delta10.txt', dtype='f4'))
    urRh_shsh_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_ur_phi90_Rh' + str(i) + '_delta10_sh_sh.txt', dtype='f4'))
    urRh_zz_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_ur_phi90_Rh' + str(i) + '_delta10.txt', dtype='f4'))
    uzRh_shsh_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_uz_phi90_Rh' + str(i) + '_delta10_sh_sh.txt', dtype='f4'))
    uzRh_zz_o.append(np.loadtxt('./data/2_nagr/2_nagr_q3_01_uz_phi90_Rh' + str(i) + '_delta10.txt', dtype='f4'))
# print(np.array(MRh_shsh_o))

MRh_shsh_o_ar = np.array(MRh_shsh_o)
MRh_zz_o_ar = np.array(MRh_zz_o)
urRh_shsh_o_ar = np.array(urRh_shsh_o)
urRh_zz_o_ar = np.array(urRh_zz_o)
uzRh_shsh_o_ar = np.array(uzRh_shsh_o)
uzRh_zz_o_ar = np.array(uzRh_zz_o)

print(np.min(MRh_shsh_o_ar[2, 16000:32000]))

for i in range(0, 4):
    MRh_shsh_extr.append(np.min(MRh_shsh_o_ar[i, 18000:30000]))
    MRh_zz_extr.append(np.min(MRh_zz_o_ar[i, 18000:30000]))
    urRh_shsh_extr.append(np.min(urRh_shsh_o_ar[i, 19000:29000]))
    urRh_zz_extr.append(np.min(urRh_zz_o_ar[i, 19000:29000]))
    uzRh_shsh_extr.append(np.min(uzRh_shsh_o_ar[i, 18000:30000]))
    uzRh_zz_extr.append(np.min(uzRh_zz_o_ar[i, 18000:30000]))

for i in range(0, 4):
    MRh_shsh_extr.append(np.min(MRh_shsh_o_ar[i, 50000:62000]))
    MRh_zz_extr.append(np.min(MRh_zz_o_ar[i, 50000:62000]))
    urRh_shsh_extr.append(np.min(urRh_shsh_o_ar[i, 51000:61000]))
    urRh_zz_extr.append(np.min(urRh_zz_o_ar[i, 51000:61000]))
    uzRh_shsh_extr.append(np.min(uzRh_shsh_o_ar[i, 50000:62000]))
    uzRh_zz_extr.append(np.min(uzRh_zz_o_ar[i, 50000:62000]))

print(MRh_shsh_extr, MRh_zz_extr)
print(urRh_shsh_extr, urRh_zz_extr)
print(uzRh_shsh_extr, uzRh_zz_extr)

# table1 = PrettyTable()
# table1.add_column('R / h', [25, 50, 100, 150])
# table1.add_column('шарнир-шарнир', [f"{MRh_shsh_extr[i]:.1e}" for i in range(0, 4)])
# table1.add_column('заделка-заделка', [f"{MRh_zz_extr[i]:.1e}" for i in range(0, 4)])
# table1.add_column('разница', [str(round(100*np.abs((MRh_shsh_extr[i] - MRh_zz_extr[i]) / MRh_shsh_extr[i]), 2)) + ' %' for i in range(0, 4)])
# print(table1)
#
# table2 = PrettyTable()
# table2.add_column('R / h', [25, 50, 100, 150])
# table2.add_column('шарнир-шарнир', [f"{MRh_shsh_extr[i]:.1e}" for i in range(4, 8)])
# table2.add_column('заделка-заделка', [f"{MRh_zz_extr[i]:.1e}" for i in range(4, 8)])
# table2.add_column('разница', [str(round(100*np.abs((MRh_shsh_extr[i] - MRh_zz_extr[i]) / MRh_shsh_extr[i]), 2)) + ' %' for i in range(4, 8)])
# print(table2)

wb = xl.Workbook('Сравнения_результатов.xlsx')
ws1 = wb.add_worksheet('Моменты')
ws2 = wb.add_worksheet('Перемещения ur')
ws3 = wb.add_worksheet('Перемещения uz')

phi_ar = [25, 50, 100, 150]
names = ['R / h', 'Шарнир-Шарнир', 'Заделка-Заделка', 'Разница']

for i in range(0, 4):
    ws1.write(0, i, names[i])
    ws1.write(6, i, names[i])

for i in range(0, 4):
    ws2.write(0, i, names[i])
    ws2.write(6, i, names[i])

for i in range(0, 4):
    ws3.write(0, i, names[i])
    ws3.write(6, i, names[i])

for i in range(0, 4):
    ws1.write(i + 1, 0, phi_ar[i])
    ws1.write(i + 7, 0, phi_ar[i])

for i in range(0, 4):
    ws2.write(i + 1, 0, phi_ar[i])
    ws2.write(i + 7, 0, phi_ar[i])

for i in range(0, 4):
    ws3.write(i + 1, 0, phi_ar[i])
    ws3.write(i + 7, 0, phi_ar[i])

for i in range(0, 4):
    ws1.write(i + 1, 1, f"{MRh_shsh_extr[i]:.2e}")
    ws1.write(i + 1, 2, f"{MRh_zz_extr[i]:.2e}")
    ws1.write(i + 1, 3, str(round(100*(np.abs(MRh_shsh_extr[i]) - np.abs(MRh_zz_extr[i])) / MRh_shsh_extr[i], 2)) + ' %')
    ws1.write(i + 7, 1, f"{MRh_shsh_extr[i + 4]:.2e}")
    ws1.write(i + 7, 2, f"{MRh_zz_extr[i + 4]:.2e}")
    ws1.write(i + 7, 3, str(round(100 * (np.abs(MRh_shsh_extr[i + 4]) - np.abs(MRh_zz_extr[i + 4])) / MRh_shsh_extr[i + 4], 2)) + ' %')

for i in range(0, 4):
    ws2.write(i + 1, 1, f"{urRh_shsh_extr[i]:.2e}")
    ws2.write(i + 1, 2, f"{urRh_zz_extr[i]:.2e}")
    ws2.write(i + 1, 3, str(round(100*(np.abs(urRh_shsh_extr[i]) - np.abs(urRh_zz_extr[i])) / urRh_shsh_extr[i], 2)) + ' %')
    ws2.write(i + 7, 1, f"{urRh_shsh_extr[i + 4]:.2e}")
    ws2.write(i + 7, 2, f"{urRh_zz_extr[i + 4]:.2e}")
    ws2.write(i + 7, 3, str(round(100*(np.abs(urRh_shsh_extr[i + 4]) - np.abs(urRh_zz_extr[i + 4])) / urRh_shsh_extr[i + 4], 2)) + ' %')

for i in range(0, 4):
    ws3.write(i + 1, 1, f"{uzRh_shsh_extr[i]:.2e}")
    ws3.write(i + 1, 2, f"{uzRh_zz_extr[i]:.2e}")
    ws3.write(i + 1, 3, str(round(100*(np.abs(uzRh_shsh_extr[i]) - np.abs(uzRh_zz_extr[i])) / uzRh_shsh_extr[i], 2)) + ' %')
    ws3.write(i + 7, 1, f"{uzRh_shsh_extr[i + 4]:.2e}")
    ws3.write(i + 7, 2, f"{uzRh_zz_extr[i + 4]:.2e}")
    ws3.write(i + 7, 3, str(round(100*(np.abs(uzRh_shsh_extr[i + 4]) - np.abs(uzRh_zz_extr[i + 4])) / uzRh_shsh_extr[i + 4], 2)) + ' %')

wb.close()












