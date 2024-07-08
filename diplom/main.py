import numpy as np
import fractions as frac
import scipy.integrate as integr
import matplotlib.pyplot as plt

# задание начальных параметров

pi = np.pi
k = 1
RR = 100
hh = 1
EU = 21000
nu = 0.3

phi0 = pi
phi1 = frac.Fraction(80, 180) * pi
delta = frac.Fraction(10, 180) * pi

q = 3
AS = phi0 - phi1 - delta
s0 = AS * RR

x = np.linspace(0, 1, 80000)

# задание функций переменных

def phi(x): return AS * x + phi1
def rr(x): return RR * np.sin(phi0 - phi(x))
def p0(x): return 1 + (hh**2 * np.sin(phi(x))**2) / (3 * rr(x)**2)
def q_r_k(x): return 0

def q_z_k(x):
    return sum(np.sin(pi*n*x / k)*2 / k*(s0 * q / (EU * hh)*np.cos(0.45*pi*n)/(pi*n) - s0 * q / (EU * hh)*np.cos(0.55*pi*n)/(pi*n)) for n in range(1, 52))

def q_theta_k(x): return 0
    
# задание правой части системы

def f(x, y):
    return [-(1-nu)*np.cos(phi(x))*s0/rr(x)*y[0]+nu*np.sin(phi(x))*s0/rr(x)*y[1]-k*np.cos(phi(x))*s0/rr(x)*y[2]+hh*k**2*nu*s0*np.sin(phi(x))/rr(x)**2*y[3]+(hh*s0)/(rr(x)**2)*(1+(k**4*hh**2*np.sin(phi(x))**2)/(12*rr(x)**2))*y[4]-k**4*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[5]+(k*hh*s0)/(rr(x)**2)*(1+(k**2*hh**2*(np.sin(phi(x)))**2)/(12*rr(x)**2))*y[6]+(k**2*hh**2*s0*np.sin(phi(x))*np.cos(phi(x)))/(12*rr(x)**3)*y[7]+q_r_k(x),
            -np.cos(phi(x))*s0/rr(x)*y[1]-k*s0*np.sin(phi(x))*(1-hh**2*np.cos(phi(x))**2/(3*rr(x)**2))/(rr(x)*p0(x))*y[2]-hh*k**2*nu*s0*np.cos(phi(x))/rr(x)**2*y[3]-k**4*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[4]+k**2*hh**3*s0*(k**2*np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**4)*y[5]-k**3*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[6]-k**2*hh**2*s0*(np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**3)*y[7]+q_z_k(x),
            k*nu*np.cos(phi(x))*s0/rr(x)*y[0]+k*nu*np.sin(phi(x))*s0/rr(x)*y[1]-2*np.cos(phi(x))*s0/rr(x)*y[2]+hh*k*nu*s0*np.sin(phi(x))/rr(x)**2*y[3]+k*hh*s0*(1+k**2*hh**2*np.sin(phi(x))**2/(12*rr(x)**2))/rr(x)**2*y[4]-k**3*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[5]+k**2*hh*s0*(1+hh**2*np.sin(phi(x))**2/(12*rr(x)**2))/rr(x)**2*y[6]+k*hh*2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[7]+q_theta_k(x),
            np.sin(phi(x))*s0/hh*y[0]-np.cos(phi(x))*s0/hh*y[1]-k*hh*np.sin(phi(x))*s0/(3*p0(x)*rr(x)**2)*y[2]-(1-nu)*np.cos(phi(x))*s0/rr(x)*y[3]+k**2*hh**2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[4]-k**2*hh**2*s0*(np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**3)*y[5]+k*hh**2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[6]+hh*s0*(np.cos(phi(x))**2+2*k**2/((1+nu)*p0(x)))/(12*rr(x)**2)*y[7],
            (-nu**2+1)*s0*np.cos(phi(x))**2/hh*y[0]+(-nu**2+1)*s0*np.sin(phi(x))*np.cos(phi(x))/hh*y[1]-nu*np.cos(phi(x))*s0/rr(x)*y[4]-nu*k*s0*np.cos(phi(x))/rr(x)*y[6]-s0*np.sin(phi(x))/hh*y[7],
            (-nu**2+1)*s0*np.sin(phi(x))*np.cos(phi(x))/hh*y[0]+(-nu**2+1)*s0*np.sin(phi(x))**2/hh*y[1]-nu*np.sin(phi(x))*s0/rr(x)*y[4]-nu*k*s0*np.sin(phi(x))/rr(x)*y[6]+s0*np.cos(phi(x))/hh*y[7],
            (2*(1+nu))*s0/(p0(x)*hh)*y[2]+k*s0*np.cos(phi(x))/rr(x)*y[4]+k*s0*np.sin(phi(x))/(p0(x)*rr(x))*(1-hh**2*np.cos(phi(x))**2/(3*rr(x)**2))*y[5]+s0*np.cos(phi(x))/rr(x)*y[6]+hh*k*s0*np.sin(phi(x))/(3*p0(x)*rr(x)**2)*y[7],
            (12*(-nu**2+1))*s0/hh*y[3]-k**2*nu*hh*s0*np.sin(phi(x))/rr(x)**2*y[4]+nu*k**2*hh*s0*np.cos(phi(x))/rr(x)**2*y[5]-k*nu*hh*s0*np.sin(phi(x))/rr(x)**2*y[6]-nu*np.cos(phi(x))*s0/rr(x)*y[7]]

# задание граничных условий

def bc(ya, yb):
    return np.array([ya[4], ya[5], ya[6], ya[7], yb[4], yb[5], yb[6], yb[7]])

# решение системы

y0 = np.zeros((8, x.size))
sol = integr.solve_bvp(f, bc, x, y0, tol=1e-8)

# вывод эпюр

for i in range(1, 9):
    fig = plt.figure(i, figsize=(7,7))
    ax = fig.add_subplot()
    plt.plot(x, sol.y[i-1], c='black')
    ax.grid(True)
    ax.set_title('Y' + str(i))

plt.show()




