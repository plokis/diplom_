import numpy as np
import fractions as frac
import scipy.integrate as integr
import matplotlib.pyplot as plt

pi = np.pi
k = 1
RR = 150
hh = 1
EU = 21000
nu = 0.5

phi0 = pi
phi1 = frac.Fraction(80, 180) * pi
delta = frac.Fraction(10, 180) * pi

q = 3
AS = phi0 - phi1 - delta
s0 = AS * RR

x = np.linspace(0, 1, 80000)
def phi(x): return AS * x + phi1
def rr(x): return RR * np.sin(phi0 - phi(x))
def p0(x): return 1 + (hh**2 * np.sin(phi(x))**2) / (3 * rr(x)**2)
def q_r_k(x): return 0

def q_z_k(x): return sum(np.sin(pi*n*x / k)*2 / k*(s0 * q / (EU * hh)*np.cos(0.45*pi*n)/(pi*n) - s0 * q / (EU * hh)*np.cos(0.55*pi*n)/(pi*n)) for n in range(1, 52))

def q_theta_k(x): return 0

def f(x, y):
    return [-(1-nu)*np.cos(phi(x))*s0/rr(x)*y[0]+nu*np.sin(phi(x))*s0/rr(x)*y[1]-k*np.cos(phi(x))*s0/rr(x)*y[2]+hh*k**2*nu*s0*np.sin(phi(x))/rr(x)**2*y[3]+(hh*s0)/(rr(x)**2)*(1+(k**4*hh**2*np.sin(phi(x))**2)/(12*rr(x)**2))*y[4]-k**4*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[5]+(k*hh*s0)/(rr(x)**2)*(1+(k**2*hh**2*(np.sin(phi(x)))**2)/(12*rr(x)**2))*y[6]+(k**2*hh**2*s0*np.sin(phi(x))*np.cos(phi(x)))/(12*rr(x)**3)*y[7]+q_r_k(x),
            -np.cos(phi(x))*s0/rr(x)*y[1]-k*s0*np.sin(phi(x))*(1-hh**2*np.cos(phi(x))**2/(3*rr(x)**2))/(rr(x)*p0(x))*y[2]-hh*k**2*nu*s0*np.cos(phi(x))/rr(x)**2*y[3]-k**4*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[4]+k**2*hh**3*s0*(k**2*np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**4)*y[5]-k**3*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[6]-k**2*hh**2*s0*(np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**3)*y[7]+q_z_k(x),
            k*nu*np.cos(phi(x))*s0/rr(x)*y[0]+k*nu*np.sin(phi(x))*s0/rr(x)*y[1]-2*np.cos(phi(x))*s0/rr(x)*y[2]+hh*k*nu*s0*np.sin(phi(x))/rr(x)**2*y[3]+k*hh*s0*(1+k**2*hh**2*np.sin(phi(x))**2/(12*rr(x)**2))/rr(x)**2*y[4]-k**3*hh**3*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**4)*y[5]+k**2*hh*s0*(1+hh**2*np.sin(phi(x))**2/(12*rr(x)**2))/rr(x)**2*y[6]+k*hh*2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[7]+q_theta_k(x),
            np.sin(phi(x))*s0/hh*y[0]-np.cos(phi(x))*s0/hh*y[1]-k*hh*np.sin(phi(x))*s0/(3*p0(x)*rr(x)**2)*y[2]-(1-nu)*np.cos(phi(x))*s0/rr(x)*y[3]+k**2*hh**2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[4]-k**2*hh**2*s0*(np.cos(phi(x))**2+2/((1+nu)*p0(x)))/(12*rr(x)**3)*y[5]+k*hh**2*s0*np.sin(phi(x))*np.cos(phi(x))/(12*rr(x)**3)*y[6]+hh*s0*(np.cos(phi(x))**2+2*k**2/((1+nu)*p0(x)))/(12*rr(x)**2)*y[7],
            (-nu**2+1)*s0*np.cos(phi(x))**2/hh*y[0]+(-nu**2+1)*s0*np.sin(phi(x))*np.cos(phi(x))/hh*y[1]-nu*np.cos(phi(x))*s0/rr(x)*y[4]-nu*k*s0*np.cos(phi(x))/rr(x)*y[6]-s0*np.sin(phi(x))/hh*y[7],
            (-nu**2+1)*s0*np.sin(phi(x))*np.cos(phi(x))/hh*y[0]+(-nu**2+1)*s0*np.sin(phi(x))**2/hh*y[1]-nu*np.sin(phi(x))*s0/rr(x)*y[4]-nu*k*s0*np.sin(phi(x))/rr(x)*y[6]+s0*np.cos(phi(x))/hh*y[7],
            (2*(1+nu))*s0/(p0(x)*hh)*y[2]+k*s0*np.cos(phi(x))/rr(x)*y[4]+k*s0*np.sin(phi(x))/(p0(x)*rr(x))*(1-hh**2*np.cos(phi(x))**2/(3*rr(x)**2))*y[5]+s0*np.cos(phi(x))/rr(x)*y[6]+hh*k*s0*np.sin(phi(x))/(3*p0(x)*rr(x)**2)*y[7],
            (12*(-nu**2+1))*s0/hh*y[3]-k**2*nu*hh*s0*np.sin(phi(x))/rr(x)**2*y[4]+nu*k**2*hh*s0*np.cos(phi(x))/rr(x)**2*y[5]-k*nu*hh*s0*np.sin(phi(x))/rr(x)**2*y[6]-nu*np.cos(phi(x))*s0/rr(x)*y[7]]
def bc(ya, yb):
    return np.array([ya[4], ya[5], ya[6], ya[7], yb[4], yb[5], yb[6], yb[7]])

y0 = np.zeros((8, x.size))
sol = integr.solve_bvp(f, bc, x, y0, tol=1e-8)

# def wrap_around(radii):
#     thetas = [i * (135 / len(radii)) for i in range(0, len(radii))]
#     xs = [((150 + radius) * np.sin(np.deg2rad(theta))) for radius, theta in zip(radii, thetas)]
#     ys = [(radius * np.cos(np.deg2rad(theta))) for radius, theta in zip(radii, thetas)]
#     print(thetas, len(thetas))
#     print(xs, len(xs))
#     print(ys, len(ys))
#     return xs,ys

# np.savetxt('q3_01_M_phi90_Rh150_delta10.txt', sol.y[3], delimiter=" ")

# Y4 = sol.y[3]

# def cart2pol(x, y):
#     rho = np.sqrt(x**2 + y**2)
#     ph = np.arctan2(y, x)
#     return rho, ph

# def pol2cart(rho, phi):
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return x, y

theta = 0.5*pi*x

for i in range(1, 9):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.plot(theta, sol.y[i-1], c='black')
    ax.set_theta_zero_location("N")
    # ax.set_rticks()
    ax.set_rlim(np.min(sol.y[i-1]), 4*np.max(sol.y[i-1]))
    ax.set_thetalim(0, 0.5*pi)
    ax.grid(True)

plt.show()