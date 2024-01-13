'''
Code to explicitly write out the Bogoliubov-de Gennes Hamiltonians
for a chain of superconducting qubits characertized by 
an on-site energy, nearest-neighbor hopping, and Cooper pairing.


'''
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from typing import Literal
import seaborn as sns

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str, choices=['kitaev', 'ssh'],
        help="Specify either Kitaev or SSH chain model.",
    )
    parser.add_argument(
        '--tune', 
        type=str, 
        choices=['mu/t', 't2/t1', 'x'], 
        help="Specify parameter to tune."
            "mu/t is ratio on-site energy to nn-hopping (kitaev only), "
            "t2/t1 is ratio of double- to single-bond hopping (ssh only), "
            "x is parameter for tuning first-last site hopping (see arg 'x' help)."
    )
    parser.add_argument(
        '-N', type=int, default=10,
        help="Number of sites (for SSH the number of C atoms will be 2x this).",
    )
    parser.add_argument(
        '-m', type=str, default='1.0',
        help="Specify onsite energy mu (arb. units).",
    )
    parser.add_argument(
        '-t', type=str, default='1.+0.j',
        help="Specify nearest-neighbor hopping (string will be evaluated and cast to complex number).",
    )
    parser.add_argument(
        '-d', type=str, default='1.+0.j',
        help="Kitaev only: Specify Cooper pairing strength."
    )
    parser.add_argument(
        '-t2', type=str, default='1.+0.j',
        help="SSH only: Specify double-bond hopping strength (unused if tuning this parameter)",
    )
    parser.add_argument(
        '-x', type=float, default=0.5,
        help="Parameter for tuning hopping linking first and last site set by t = (1-2x) * t0 "
            "where t0 is NN-hopping (Kitaev) or single-bond hopping (SSH)",
    )
    parser.add_argument(
        '--enforce-even-sites', action='store_true', default=False,
        help="In SSH model enforce an even number of C-atom sites (odd number of hopping terms).",
    )
    return parser.parse_args(args)


class H_BdG_constructor():
    '''
    Construct the Bogoliubov-de Gennes Hamiltonian for either a Kitaev chain or SSH chain model.
    Plot the energies for the system as a given parameter is tuned.
    
    When initiating:
        1st positional arg: 'Kitaev' or 'SSH'
        Positional args: N_sites:int=5 #
    '''
    def __init__(self,
        model:Literal['kitaev', 'ssh'],
        param_label:str,
        param_space:np.array,
        **kwargs,
    ):
        self.model = model
        self.param_label = param_label
        self.param_space = param_space
        self.N_sites = kwargs.get('N')

        if self.model.lower() == 'kitaev':
            # Kitaev model is 1d chain of superconducing qubits characterized by parameters t, m, d
            # that specify nn-hopping strength, on-site energy, cooper pairing strength, respectively
            self.t = kwargs.get('t_hopping', 1.+0.j) 
            self.m = kwargs.get('mu_onsite', 1.+0.j)
            self.d = kwargs.get('d_cooper', 1.+0.j)
            self.h_size = 2 * N

            self.descr = f"Kitaev t: {self.t}, d: {self.d}"
            if 'mu' not in self.param_label:
                self.descr += f", mu: {self.m}"

        elif self.model.lower() == 'ssh':
            # SSH model is polyacetylene chain characterized by t1 and t2,
            # that specify the hopping along a single and double bond, respectively
            self.t1 = kwargs.get('t1', 1.+0.j) 
            self.t2 = kwargs.get('t2', 1.+0.j)
            # For ssh actual number of atomic C sites is kinda ambiguous
            # so let user choose to enforce an even number or not
            self.enforce_even_sites = kwargs.get('enforce_even_sites', False)
            if self.enforce_even_sites:
                self.h_size = 2 * N
            else:
                self.h_size = 2 * N + 1

            self.descr = f"SSH t1: {self.t1}"
            if 't2' not in self.param_label:
                self.descr += f", t2: {self.t2}"
            if self.enforce_even_sites:
                self.descr += f", even sites"


    def construct_and_solve_hamiltonians(self):
        # Construct and solve H_BdG across parameter space
        self.energy_bands = []
        self.E0_state_densities = []
        self.E1_state_densities = []
        for param in self.param_space:
            if self.model == 'kitaev':
                if self.param_label == 'x':
                    # Closed-loop Kitaev chain, tune loop-closing strength
                    H_BdG = construct_hamiltonian_kitaev(
                        self.N_sites, self.m, self.t, self.d, 
                        last_first_hop=1.*param
                    )
                else:
                    # Open-loop Kitaev chain, tune mu/t
                    H_BdG = construct_hamiltonian_kitaev(
                        self.N_sites, param*self.t, self.t, self.d, 
                    )
            elif self.model == 'ssh':
                if self.param_label == 'x':
                    # Closed-loop SSH chain, tune loop-closing strength
                    H_BdG = construct_hamiltonian_ssh(
                        self.N_sites, self.t1, self.t2, 
                        close_loop_hopping=1.*param,
                        even_sites=self.enforce_even_sites
                    )
                else:
                    # Open-loop SSH chain, tune t2/t1
                    H_BdG = construct_hamiltonian_ssh(
                        N, self.t1, param*self.t1,
                        close_loop_hopping=None,
                        even_sites=self.enforce_even_sites,
                    )
            energies, E0_density, E1_density = self.get_evals_and_state_densities(H_BdG)
            # Save energies and electron densities to object
            self.energy_bands.append(sorted(energies))
            self.E0_state_densities.append(E0_density)
            self.E1_state_densities.append(E1_density)
            
            if param == self.param_space[0]:
                _, axs = plt.subplots(1,2)
                sns.heatmap(ax=axs[0], data=np.real(H_BdG))
                axs[0].set_title('real')
                sns.heatmap(ax=axs[1], data=np.imag(H_BdG))
                axs[1].set_title('imag')
                plt.suptitle('H_BdG @ param[0]')
                plt.show()

    def get_evals_and_state_densities(self, H_BdG):
        '''
        Given a BdG Hamiltonian, calculate the energies.
        Also return the wavefunction density for the most- and second-most weakly bound state.
        '''

        evals, evecs = np.linalg.eigh(H_BdG)

        # Number of physical sites is half the number of quasiparticle energies
        n = int(len(evals) / 2)

        # Get wavefunctions for the lowest- and next-lowest-energy states
        E0_idx = np.where(np.isin(np.abs(evals), sorted(np.abs(evals))[:2]))[0]
        E1_idx = np.where(np.isin(np.abs(evals), sorted(np.abs(evals))[2:4]))[0]

        E0_states = (evecs[:, E0_idx[0]], evecs[:, E0_idx[1]])
        E1_states = (evecs[:, E1_idx[0]], evecs[:, E1_idx[1]])

        E0_state_density = np.abs(E0_states[0][:n])**2+ np.abs(E0_states[0][n:]) ** 2
        E0_state_density += np.abs(E0_states[1][:n])**2+ np.abs(E0_states[1][n:]) ** 2
        
        E1_state_density = np.abs(E1_states[0][:n])**2 + np.abs(E1_states[0][n:]) ** 2
        E1_state_density += np.abs(E1_states[1][:n])**2 + np.abs(E1_states[1][n:]) ** 2

        return sorted(evals), E0_state_density, E1_state_density

    def plot_figures(self):
        # Plot the energy spectrum across parameter space
        plt.subplot(121)
        plt.title('Energies (arb. units)')
        plt.xlabel(self.param_label)
        for i in range(len(self.energy_bands[0])):
            plt.plot(self.param_space, [x[i] for x in self.energy_bands], 'k')

        # Plot wavefunction densities (across sites) at a few parameter values
        param_idxs = [0, int(len(self.param_space)/2), len(self.param_space)-1]
        subplot_idxs = [2,4,6]
        for param_idx, subplot_idx in zip(param_idxs, subplot_idxs):
            plt.subplot(3,2,subplot_idx)
            plt.title(f'Wavefunction density at {self.param_label}={round(self.param_space[param_idx],1)}')
            plt.xlabel('Atomic site number')
            plt.plot(self.E0_state_densities[param_idx], 'b', label='0')
            plt.plot(self.E1_state_densities[param_idx], 'r', label='1')
        
        plt.suptitle(self.descr)
        plt.tight_layout()
        plt.legend()
        plt.show()


def construct_hamiltonian_kitaev(N, mu_onsite, t_nn, d_cooper, last_first_hop=None):
    '''
    N is number of atomic sites
    mu_onsite is onsite energy, t_nn is nearest-neighbor hopping, d_cooper is Cooper pairing strength

    boundary_hopping sets the strength of hopping t* between first and last site (closing the loop).
    The value of boundary_hopping is the x in t* = t_nn(1 - 2x) (knob to vary t* between -t_nn and t_nn)
    '''
    block_size = (N,)*2
    # Create the H block of H_BdG
    H_block = np.zeros(block_size, dtype=np.cdouble)
    # Add on-site and nn-hopping terms
    for i in range(H_block.shape[0] - 1):
        H_block[i,i] += mu_onsite
        H_block[i, i+1] += t_nn.conjugate()
        H_block[i+1, i] += t_nn
    H_block[-1,-1] += mu_onsite

    if last_first_hop is not None:
        # Add close-loop hopping from first and last site
        H_block[-1, 0] += t_nn * (1. - 2. * last_first_hop) 
        H_block[0, -1] += (t_nn * (1. - 2. * last_first_hop)).conjugate()


    # Create the (upper-right) D-block of H_BdG
    D_block = np.zeros(block_size, dtype=np.cdouble)
    # Add the Cooper-pairing terms
    for i in range(D_block.shape[0] - 1):
        D_block[i, i+1] += -d_cooper.conjugate()
        D_block[i+1, i] += d_cooper.conjugate()

    # Create final H_BdG by stitching together the H- and D-blocks with their negative conjugates
    H_top = np.hstack((H_block, D_block))
    H_bottom = np.hstack((-D_block.conjugate(), -H_block.conjugate()))
    H_BdG = np.vstack((H_top, H_bottom))
    
    return H_BdG

def construct_hamiltonian_ssh(N, t1, t2, close_loop_hopping=None, even_sites=False):
    '''
    Construct Bogoliubov-de Gennes Hamiltonian for SSH model (polyacetylene chain)
    t1 and t2 are the two hopping strengths electron encounters as it hops along dimer chain
    '''
    # Create the H block of H_BdG
    if not even_sites:
        # Option to enforce an even number of C-atom sites (odd number of hopping terms)
        block_size = (2 * N + 1, ) * 2
    else:
        # By default, number of C-atom sites is odd (even number of hopping terms)
        block_size = (2 * N, ) * 2

    H_block = np.zeros(block_size, dtype=np.cdouble)
    # Add t1-hopping terms
    for i in range(0, N):
        H_block[2*i, 2*i+1] += t1
        if i < N-1 or not even_sites:
            H_block[2*i+1, 2*i+2] += t2
    
    if close_loop_hopping is not None:
        # Add close-loop hopping from first and last site
        H_block[-1, 0] += t2 * (1.0 - 2.0 * close_loop_hopping)
    H_block = H_block + H_block.T.conjugate()
    
    # Create the D block of H_BdG
    # SSH chain is not superconducting so the off-diag subblocks of BdG Ham are empty
    D_block = np.zeros(block_size, dtype=np.cdouble)

    # Create final H_BdG by stitching together the H- and D-blocks with their negative conjugates
    H_top = np.hstack((H_block, D_block))
    H_bottom = np.hstack((-D_block.conjugate(), -H_block.conjugate()))
    H_BdG = np.vstack((H_top, H_bottom))
    
    return H_BdG


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    N = args.N # Number of unit cells
    param_label = args.tune # Parameter to tune
    close_loop_param = args.x 

    if args.model.lower() == 'kitaev':
        # Get Kitaev model-defining parameters
        t_hopping = complex(eval(args.t))
        d_cooper = complex(eval(args.d))
        mu_onsite = complex(eval(args.m)) # Unused if tuning mu/t
        num_sites = N

        # Set up parameter space to tune system through
        if param_label is None or param_label == 'mu/t':
            param_label = 'mu/t'
            param_space = np.linspace(0.0, 3.0, 41)
        elif param_label == 'x':
            param_space = np.linspace(0., 1., 41) 


        # Get description for plots
        model_descr = f"Kitaev t: {t_hopping}, d: {d_cooper}"
        if 'mu' not in param_label:
            model_descr += f", mu: {mu_onsite}"

        # Initiate the H_BdG-constructing class
        H_constructor = H_BdG_constructor(
            'kitaev', param_label, param_space, N=N,
            t_hopping=t_hopping, d_cooper=d_cooper, mu_onsite=mu_onsite,
        )

    elif args.model.lower() == 'ssh':
        # Get SSH model-defining parameters
        t1_hopping = 1.0 + 0.0j # Reference strength for SSH single-bond hopping
        t2_hopping = complex(eval(args.t2)) # Not used if t2 is tuning parameter
        num_sites = 2 * N # 

        # Set up param space to tune system through
        # Currently all param share same range for values of interest
        if param_label is None:
            param_label = 't2/t1'
        param_space = np.linspace(0., 1., 41)
        
        # Get description for plots
        model_descr = f"SSH t1: {t1_hopping}"
        if 't2' not in param_label:
            model_descr += f", t2: {t2_hopping}"
        if not args.enforce_even_sites:
            num_sites = 2 * N + 1
            model_descr += ', even sites'

        # Initiate Hamiltonian constructor class
        H_constructor = H_BdG_constructor(
            'ssh', param_label, param_space, N=N,
            t1_hopping=t1_hopping, t2_hopping=t2_hopping,
            enforce_even_sites=args.enforce_even_sites,
        )

    H_constructor.construct_and_solve_hamiltonians()
    H_constructor.plot_figures()

