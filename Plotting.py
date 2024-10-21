## Plot Histograms Main Project
import numpy as np
import matplotlib.pyplot as plt
import pickle
#import sys
#sys.path.append('C:\\Users\\safie\\OneDrive\\')


J = 10
num_particles = 1000

objects = []
with (open("Master Thesis - Standard Posterior.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

Y = objects[0]
particles = objects[1]
wghts = objects[2]
eta = objects[3]

colors = ['red','teal','goldenrod','orchid','wheat','darkgreen','aquamarine','crimson','orange','silver','plum','lightblue','lavender','lightgreen','pink','coral','khaki','violet','sienna','indigo']

print('True parameters:')
print(Y)
print('Empirical mean:')
print(np.mean(particles,axis=0))
print('Emperical variance:')
print(np.var(particles,axis=0))
for i in range(1,2*J+1):
    plt.subplot(5,4,i)
    plt.hist(particles[:,i-1],weights=wghts,color=colors[i-1])
    plt.title('$Y_{{{}}}$ = {:.5f}'.format(i,Y[i-1]))
    plt.vlines(Y[i-1],-0.01,0.5)
    plt.grid()
    plt.xlim((-1,1))

plt.show()


## Plot Radii Main Project
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
import pickle


# (Usually) Fixed Parameters
M       = 1000
r0      = 0.01
epsilon = 0.001

# Variable Parameters
s_1   = 0.1
s_2   = 0.001
sum_1 = 4.124
sum_2 = 44.1466
J_1   = 6
J_2   = 58

objects_prior_1 = []
objects_prior_2 = []
objects_post_1  = []
objects_post_2  = []

with (open("Master Thesis - s=0.1 Prior.pickle", "rb")) as openfile:
    while True:
        try:
            objects_prior_1.append(pickle.load(openfile))
        except EOFError:
            break

with (open("Master Thesis - s=0.001 Prior.pickle", "rb")) as openfile:
    while True:
        try:
            objects_prior_2.append(pickle.load(openfile))
        except EOFError:
            break

with (open("Master Thesis - s=0.1 Posterior.pickle", "rb")) as openfile:
    while True:
        try:
            objects_post_1.append(pickle.load(openfile))
        except EOFError:
            break

with (open("Master Thesis - s=0.001 Posterior.pickle", "rb")) as openfile:
    while True:
        try:
            objects_post_2.append(pickle.load(openfile))
        except EOFError:
            break

Y_1               = objects_prior_1[0]
Y_2               = objects_prior_2[0]
particles_prior_1 = objects_prior_1[1]
weights_prior_1   = objects_prior_1[2]
particles_prior_2 = objects_prior_2[1]
weights_prior_2   = objects_prior_2[2]
particles_post_1  = objects_post_1[1]
weights_post_1    = objects_post_1[2]
particles_post_2  = objects_post_2[1]
weights_post_2    = objects_post_2[2]

EY_prior_1  = np.dot(weights_prior_1.T,particles_prior_1)
EY_prior_2  = np.dot(weights_prior_2.T,particles_prior_2)
EY_post_1   = np.dot(weights_post_1.T, particles_post_1)
EY_post_2   = np.dot(weights_post_2.T, particles_post_2)
VarY_post_1 = np.dot(weights_post_1.T, (particles_post_1-EY_post_1*np.ones((M,2*J_1)))**2)
VarY_post_2 = np.dot(weights_post_2.T, (particles_post_2-EY_post_2*np.ones((M,2*J_2)))**2)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def radius_scatt(Y,s,sum,J):
    r = r0*np.ones(len(phi))
    for j in range(1,J+1):
        #r += r0/(4*zeta(2+epsilon)*j**(2+epsilon))*(Y[2*j-2]*np.cos(j*phi)+Y[2*j-1]*np.sin(j*phi))
        r += r0/(4*sum*(1+s*j**(2+epsilon)))*(Y[2*j-2]*np.cos(j*phi)+Y[2*j-1]*np.sin(j*phi))
    return r

def radius_var_scatt(Y,s,sum,J):
    r = np.zeros(len(phi))
    for j in range(1,J+1):
        #r += (r0/(4*zeta(2+epsilon)*j**(2+epsilon)))**2*(Y[2*j-2]*np.cos(j*phi)**2+Y[2*j-1]*np.sin(j*phi)**2)
        r += (r0/(4*sum*(1+s*j**(2+epsilon))))**2*(Y[2*j-2]*np.cos(j*phi)**2+Y[2*j-1]*np.sin(j*phi)**2)
    return r

phi        = np.linspace(0,2*np.pi,1000)
r_1        = radius_scatt(Y_1,s_1,sum_1,J_1)
r_2        = radius_scatt(Y_2,s_2,sum_2,J_2)
Er_prior_1 = radius_scatt(EY_prior_1,s_1,sum_1,J_1)
Er_prior_2 = radius_scatt(EY_prior_2,s_2,sum_2,J_2)
Er_post_1  = radius_scatt(EY_post_1 ,s_1,sum_1,J_1)
Er_post_2  = radius_scatt(EY_post_2 ,s_2,sum_2,J_2)

Varr_post_1  = radius_var_scatt(VarY_post_1,s_1,sum_1,J_1)
Varr_post_2  = radius_var_scatt(VarY_post_2,s_2,sum_2,J_2)
r_min_post_1 = Er_post_1-np.sqrt(Varr_post_1)
r_max_post_1 = Er_post_1+np.sqrt(Varr_post_1)
r_min_post_2 = Er_post_2-np.sqrt(Varr_post_2)
r_max_post_2 = Er_post_2+np.sqrt(Varr_post_2)

x_1,y_1                   = pol2cart(r_1,phi)
x_2,y_2                   = pol2cart(r_2,phi)
Ex_prior_1,Ey_prior_1     = pol2cart(Er_prior_1,phi)
Ex_prior_2,Ey_prior_2     = pol2cart(Er_prior_2,phi)
Ex_post_1 ,Ey_post_1      = pol2cart(Er_post_1, phi)
Ex_post_2 ,Ey_post_2      = pol2cart(Er_post_2, phi)
x_min_post_1,y_min_post_1 = pol2cart(r_min_post_1,phi)
x_max_post_1,y_max_post_1 = pol2cart(r_max_post_1,phi)
x_min_post_2,y_min_post_2 = pol2cart(r_min_post_2,phi)
x_max_post_2,y_max_post_2 = pol2cart(r_max_post_2,phi)

plt.subplot(1,2,1)
plt.plot(x_1       ,y_1       ,label='$\hat{r}(\phi)$')
plt.plot(Ex_prior_1,Ey_prior_1,label='$E_{prior}[r(\phi)]$')
plt.plot(Ex_post_1 ,Ey_post_1 ,label='$E_{post}[r(\phi)]$')
plt.fill(np.append(x_min_post_1,x_max_post_1[::-1]),np.append(y_min_post_1,y_max_post_1[::-1]),color='grey',label='$\sigma$-inverval')
plt.legend()
plt.grid()
plt.title('Prior and Posterior Expectation for $s=0.1$')

plt.subplot(1,2,2)
plt.plot(x_2       ,y_2       ,label='$\hat{r}(\phi)$')
plt.plot(Ex_prior_2,Ey_prior_2,label='$E_{prior}[r(\phi)]$')
plt.plot(Ex_post_2 ,Ey_post_2 ,label='$E_{post}[r(\phi)]$')
plt.fill(np.append(x_min_post_2,x_max_post_2[::-1]),np.append(y_min_post_2,y_max_post_2[::-1]),color='grey',label='$\sigma$-inverval')
plt.legend()
plt.grid()
plt.title('Prior and Posterior Expectation for $s=0.001$')

plt.show()
