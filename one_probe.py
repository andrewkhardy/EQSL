'''
_____________________________________________________
|                                                    |
|              2- Pump Probe Dynamics                |
|  For a 3-Level- Superconducting circuit and cavity |
|    Written by Helen Percival and Andrew Hardy      |
|                                                    |
|          Last Updated August 3rd 2019              |
|____________________________________________________|
'''
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

def ham( delta_c, delta_p, Omega_c, Omega_p, chi, kappa,gamma):
	""" This creates the Hamiltonian and the decay operators
	Args:
	----------
	delta_c (float) : detuning of the coupling beam
	delta_p (float): detuning of the probe beam
	Omega_c (float): Rabi Frequency of coupling beam
	Omega_p (float): Rabi Frequency of probe beam
	chi (float) : dispersive shift, or cavity pull
	kappa (float) :  Decay rate of the cavity
	gamma (float) : Decay rate of the qubit
	 Returns:
	 	H (2D array, QuTiP object):  The Hamiltonian
	 	c_op_list (list) : A list of 2 QuTip objects, the decay operators
	----------
	"""
	n = 3
	g = basis(2, 0)
	e = basis (2, 1)
	#
	H = \
	n*(-delta_p -2*chi)*tensor(g*g.dag(), num(n)) \
	- delta_c   * tensor(e*e.dag(),qeye(n))  \
	- delta_p   * tensor(e * e.dag(), num(n)) \
	+ delta_p   * tensor(e*e.dag(),qeye(n))    \
	+ Omega_c/2 * tensor(create(2),create(n))   \
	+ Omega_p/2 * tensor(e*e.dag(),create(n))    \
	+ Omega_p/2 * tensor(e*e.dag(),destroy(n))    \
	+ Omega_p/2 * tensor(g*g.dag(), create(n))     \
	+ Omega_p/2 * tensor(g*g.dag(), destroy(n))     \
	+ Omega_c/2 * tensor(destroy(2), destroy(n))
	# equations to simulate loss

	p_1 = tensor(qeye(2),destroy(n)) * np.sqrt(kappa)
	p_2 = tensor(destroy(2), qeye(n)) * np.sqrt(gamma)
	return H, [p_1,p_2,]
def run():
	# define fitting parameters
	N = 100
	M = 100
	p_tot = 5
	c_tot = 5
	# define excitation population
	e_pop = np.zeros([M,N])
	chi = -1.05*2*np.pi  # Dispersive shift in cavity resonance frequency due to qubit state
	delta_p = np.linspace(-4*np.pi,8*np.pi,N+1)  # Probe frequency detuning
	delta_c = np.linspace(-8*np.pi,8*np.pi,M+1)  # Coupler frequency detuning
	kappa = .25*2*np.pi       # 2.9                   # Decay rate of the cavity
	gamma = 0.0625*2*np.pi     # 0.04                    # Decay rate of the qubit
	n = 3
	for number in range(1,n):
		for l in range(2):
			for p in range(1,p_tot):
				for c in range(1, c_tot):
					Omega_p = .15*p*2*np.pi                        # Probe Rabi drive frequency
					Omega_c = 0.5*c*2*np.pi                        # Coupler Rabi drive frequency
					# making num operator
					level = basis(2,l)
					level_op = level*level.dag()
					num_op = number*basis(n,number)*basis(n,number).dag()
					expector = tensor(level_op,num_op)
					# time stepping
					for i in range(N):
						print(i)
						for j in range(M):
							H, c_op_list = ham(delta_c[j], delta_p[i], Omega_c, Omega_p, chi, kappa, gamma)
							result = steadystate(H, c_op_list)
							e_pop[j,i] = expect(expector, result)
					# naming code
					if l == 0:
						state = 'g'
					else:
						state = 'e'
					# plotting functionality
					fig, ax = plt.subplots( figsize = (10 ,10))
					#np.savetxt('detuning_Ap5_Oc400_test.txt' ,e_pop, fmt= '%f' , delimiter= ',' )
					plot = ax.pcolor(delta_p/(2*np.pi),delta_c/(2*np.pi), e_pop, edgecolors= 'none' )
					plot.set_cmap ('RdYlBu_r')
					ax.set_ylabel(r'$\Delta_{c}/2\pi$', fontsize=20)
					ax.set_xlabel(r'$\Delta_{p}/2\pi$', fontsize=20)
					ax.axis('tight')
					ax.set_title('Population of '+ state +'n = '+np.str(number), fontsize=16)
					plt.colorbar(plot)
					plt.savefig(state+np.str(number)+np.str(p)+np.str(c)+'.png')
					#plt.show()
					print('done a round')
# running
if __name__ == '__main__':
    run()
