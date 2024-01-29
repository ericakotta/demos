'''
Code to explicitly write out the Bogoliubov-de Gennes Hamiltonians
for a chain of superconducting qubits characertized by 
an on-site energy, nearest-neighbor hopping, and Cooper pairing.


'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Literal
import seaborn as sns

'''hello'''

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
        '-N', type=int, default=5,
        help="Number of sites (for SSH the number of C atoms will be 2x this).",
    )
    parser.add_argument(
        '-m', type=float, nargs='+', default=[1.],
        help="Specify onsite energy mu (arb. units). Specify two numbers if complex, e.g. '0.1 0.5' for .1+.5j",
    )
    parser.add_argument(
        '-t', type=float, nargs='+', default=[1.],
        help="Specify nearest-neighbor hopping (string will be evaluated and cast to complex number).",
    )
    parser.add_argument(
        '-d', type=float, nargs='+', default=[1.],
        help="Kitaev only: Specify Cooper pairing strength."
    )
    parser.add_argument(
        '-t2', type=float, nargs='+', default=[1.],
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
    parser.add_argument(
        '--plot-band-idxs', type=int, nargs='+', default=[0, 1], 
        help="Enter idx of the energy modes (0=lowest-energy, etc) to see spatial distribution of"
    )
    parser.add_argument(
        '--show-ham', action='store_true', default=False,
        help="Show heatmap of BdG Hamiltonian at the firs parameter value as sanity check.",
    )
    return parser.parse_args(args)


class H_BdG_constructor():
    '''
    Construct the Bogoliubov-de Gennes Hamiltonian for either a Kitaev chain or SSH chain model.
    Plot the energies for the system as a given parameter is tuned.
    
    When initiating:
        positional arg model: 'Kitaev' or 'SSH'
        positional arg model_params: dict of model-defining parameters

    '''
    def __init__(self,
        model:Literal['kitaev', 'ssh'],
        model_params:dict,
        **kwargs,
    ):
        self.model = model
        self.model_params = model_params

        self.param_label = model_params.get('tuning_parameter', 'mu/t')
        self.param_space = model_params.get('parameter_space', np.linspace(0, 1, 51))
        self.N_sites = model_params.get('N_sites', 5)
        self.plot_band_idxs = kwargs.get('plot_band_idxs', [0,1])
        self.show_ham = kwargs.get('show_ham', False)

        if self.model.lower() == 'kitaev':
            # Kitaev model is 1d chain of superconducing qubits characterized by parameters t, m, d
            # that specify nn-hopping strength, on-site energy, cooper pairing strength, respectively
            self.t = model_params.get('t_hopping', 1.0)
            self.m = model_params.get('m_onsite', 1.0)
            self.d = model_params.get('d_cooper', 1.0)
            self.x = model_params.get('x_periodic', 0.5)

            self.descr = f"Kitaev t: {self.t}, d: {self.d}"
            if self.param_label.lower() == 'x':
                self.descr += f", m: {self.m}"
                self.ROUTINE = 1
                print(f"Constructing BdG Hamiltoninan for Kitaev chain tuning N-to-1 hopping...")

            else:
                self.descr += f", x: {self.x}"
                self.ROUTINE = 0
                print(f"Constructing BdG Hamiltonian for Kitaev chain tuning on-site energy...")

        elif self.model.lower() == 'ssh':
            # SSH model is polyacetylene chain characterized by t1 and t2,
            # that specify the hopping along a single and double bond, respectively
            self.t1 = model_params['t1_hopping']
            self.t2 = model_params['t2_hopping']

            # For ssh actual number of atomic C sites is kinda ambiguous
            # so let user choose to enforce an even number or not
            self.enforce_even_sites = kwargs.get('enforce_even_sites', False)
            if self.enforce_even_sites:
                pass
            
            self.descr = f"SSH t1: {self.t1}"
            if 't2' not in self.param_label:
                self.descr += f", t2: {self.t2}"
            if self.enforce_even_sites:
                self.descr += f", even sites"
            
            if self.param_label == 'x':
                self.ROUTINE = 3
                print(f"Constructing BdG Hamiltonian for SSH chain tuning N-to-1 hopping...")
            else:
                self.ROUTINE = 2
                print(f"Constructing BdG Hamiltonian for SSH chain tuning t2 hopping strength... ")


    def construct_and_solve_hamiltonians(self):
        # Construct and solve H_BdG across parameter space
        self.energy_bands = []
        self.plot_state_densities = []
        for param in self.param_space:
            # Evaluate H_BdG at each parameter increment 
            if self.ROUTINE == 0:
                # Open-loop Kitaev chain, tune mu/t
                H_BdG = construct_hamiltonian_kitaev(
                    self.N_sites, param*self.t, self.t, self.d, x_periodic=self.x,
                )
            elif self.ROUTINE == 1:
                H_BdG = construct_hamiltonian_kitaev(
                    self.N_sites, self.m, self.t, self.d, x_periodic=1.*param,
                )
            elif self.ROUTINE == 2:
                H_BdG = construct_hamiltonian_ssh(
                    self.N_sites, self.t1, param*self.t1,
                    close_loop_hopping=None,
                    even_sites=self.enforce_even_sites,
                )
            elif self.ROUTINE == 3:
                H_BdG = construct_hamiltonian_ssh(
                    self.N_sites, self.t1, self.t2, 
                    close_loop_hopping=1.*param,
                    even_sites=self.enforce_even_sites
                )
            
            energies, plot_state_densities = self.get_evals_and_state_densities(H_BdG, see_idx=self.plot_band_idxs)
            # Save energies and electron densities to object
            self.energy_bands.append(sorted(energies))
            self.plot_state_densities.append(plot_state_densities)
            
            if self.show_ham and param == self.param_space[0]:
                _, axs = plt.subplots(1,2, figsize=(10,4))
                sns.heatmap(ax=axs[0], data=np.real(H_BdG), annot=False, square=True)
                axs[0].set_title('real')
                sns.heatmap(ax=axs[1], data=np.imag(H_BdG), annot=False, square=True)
                axs[1].set_title('imag')
                plt.suptitle(f'H_BdG @ {self.param_label}={param}')
                plt.show()

    def get_evals_and_state_densities(self, H_BdG, see_idx=[0,1]):
        '''
        Given a BdG Hamiltonian, calculate the energies.
        Also return the wavefunction density for the most- and second-most weakly bound state.
        '''
        evals, evecs = np.linalg.eig(H_BdG)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:,idx]

        # Number of physical sites is half the number of quasiparticle energies
        n = int(len(evals) / 2)

        # Get wavefunctions for the specified energy states
        plot_state_densities = []
        for idx in see_idx:
            state1, state2 = evecs[:, n + idx], evecs[:, n -idx - 1]
            state_density1 = np.abs(state1[:n])**2+ np.abs(state1[n:]) ** 2
            state_density2 = np.abs(state2[:n])**2+ np.abs(state2[n:]) ** 2
            plot_state_densities.extend([state_density1, state_density2])

        return evals, plot_state_densities

    def plot_figures(self, param_values=None, figsize=(10,7)):
        # Plot the energy spectrum across parameter space
        if param_values is None:
            param_idxs = [0, int(len(self.param_space)/2), len(self.param_space)-1]
            subplot_idxs = [2,4,6]
        else:
            param_idxs = []
            subplot_idxs = []
            for idx, param in enumerate(param_values):
                ds = np.abs(np.array(self.param_space) - param)
                param_idx = np.where(ds == min(ds))[0][0]
                param_idxs.append(param_idx)
                subplot_idxs.append(idx*2 + 2)

        fig, axs = plt.subplots(3,2, figsize=figsize)
        plt.clf()
        plt.subplot(121)
        plt.title('Energies (arb. units)')
        plt.xlabel(self.param_label)
        for i in range(len(self.energy_bands[0])):
            plt.plot(self.param_space, [np.real(x[i]) for x in self.energy_bands], 'k')

        # Plot wavefunction densities (across sites) at a few parameter values
        N = len(self.plot_state_densities[0])
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.turbo(np.linspace(0,1,N)))
        markers = ['v','^']
        for param_idx, subplot_idx in zip(param_idxs, subplot_idxs):
            
            plt.subplot(len(param_idxs),2,subplot_idx)#plt.subplot(3,2,subplot_idx)
            plt.title(f'State density at {self.param_label}={round(self.param_space[param_idx],3)}', fontsize=8)
            plt.xlabel('Atomic site number')
            state_densities = self.plot_state_densities[param_idx]
            for i, state_densities in enumerate(state_densities):
                plt.plot(state_densities, marker=f'${i}$', alpha=0.5, label=f'{i}')
            if subplot_idx in subplot_idxs[:-1]:
                plt.gca().set_xticklabels([])
                plt.gca().set_xlabel('')
            if subplot_idx == subplot_idxs[0]:
                plt.gca().legend(title='State idx', fontsize=7, bbox_to_anchor=(1.02, 1), loc=0, borderaxespad=0)
        plt.suptitle(self.descr)
        plt.show()


def construct_hamiltonian_kitaev(
        N, mu_onsite, t_nn, d_cooper, x_periodic=0.5,
    ):
    '''
    N is number of atomic sites
    mu_onsite is onsite energy, t_nn is nearest-neighbor hopping, d_cooper is Cooper pairing strength

    boundary_hopping sets the strength of hopping t* between first and last site (closing the loop).
    The value of boundary_hopping is the x in t* = t_nn(1 - 2x) (knob to vary t* between -t_nn and t_nn)
    '''
    # Get on-site energy contribution
    Hm = np.zeros((2*N,)*2, dtype=np.complex128)
    for i in range(N):
        Hm[i,i] += -mu_onsite
        Hm[i+N, i+N] += mu_onsite

    # Get nn-hopping contribution
    Ht = np.zeros((2*N,)*2, dtype=np.complex128)
    for i in range(N - 1):
        Ht[i, i+1] += -t_nn
        Ht[i+1, i] += -t_nn
        Ht[N+i, N+i+1] += t_nn
        Ht[N+i+1, N+i] += t_nn

    # Get cooper-pairing contribution
    Hd = np.zeros((2*N,)*2, dtype=np.complex128)
    for i in range(N -1):
        Hd[i, N+i+1] += d_cooper
        Hd[i+1, N+i] += -d_cooper
        Hd[i+N, i+1] += -d_cooper
        Hd[i+N+1, i] += d_cooper

    if x_periodic is not None:
        # Add hopping between last and first site
        phase = 1 - 2 * x_periodic#np.cos(x_periodic * np.pi )
        Ht[0, N-1] += t_nn * phase
        Ht[N-1, 0] += t_nn * phase 
        Ht[N, -1] += -t_nn * phase
        Ht[-1, N] += -t_nn * phase
        # Allow Cooper pairing between last and first site of chain
        Hd[N, N-1] += -d_cooper * phase# (2 * x_periodic - 1)
        Hd[-1, 0] += d_cooper * phase# (2 * x_periodic - 1)
        Hd[0, -1] += d_cooper * phase#(2 * x_periodic - 1)
        Hd[N-1, N] += -d_cooper * phase#(2 * x_periodic - 1)  

    return Hm + Ht + Hd


def construct_hamiltonian_ssh(
        N, t1, t2, 
        close_loop_hopping=None,
        even_sites=False,
    ):
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


def parse_complex_arg(arg):
    '''Get complex value from a list of strings encoding the real and imaginary parts'''
    if len(arg) == 1:
        return float(arg[0])
    elif len(arg) == 2:
        return arg[0] + 1j * arg[1]
    else:
        sys.exit(f"Couldn't parse an input complex number: {arg}")


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    param_label = args.tune # Parameter to tune

    if param_label == 'x':
        param_space = np.linspace(0., 1., 41)

    # Get model-defining parameters
    if args.model.lower() == 'kitaev':
        # Set up parameter space to tune system through
        if param_label is None or param_label == 'mu/t':
            param_label = 'mu/t'
            param_space = np.linspace(0.01, 4.0, 101)

        model_params = {
            'm_onsite': parse_complex_arg(args.m), # Unused if tuning mu/t
            'd_cooper': parse_complex_arg(args.d), # Cooper pairing strength
            't_hopping': parse_complex_arg(args.t), # nearest-neighbor hopping
            'x_periodic': float(args.x), # last-to-first periodic hopping
            'N_sites': args.N,
            'parameter_space': param_space,
            'tuning_parameter': param_label,
        }

    elif args.model.lower() == 'ssh':
        # Set up param space to tune system through
        if param_label is None:
            param_label = 't2/t1'
            param_space = np.linspace(0, 1.0, 101)
        
        model_params = {
            't1_hopping': 1.0 + 0.0j , # Reference strength for SSH single-bond hopping
            't2_hopping': parse_complex_arg(args.t2), # Not used if t2 is tuning parameter
            'N_sites': 2 * args.N,
            'parameter_space': param_space,
            'tuning_parameter': param_label,
        }

    # Initiate Hamiltonian constructor class
    H_constructor = H_BdG_constructor(
        args.model,
        model_params,
        enforce_even_sites=args.enforce_even_sites,
        plot_band_idxs=args.plot_band_idxs,
        show_ham=args.show_ham,
    )
    H_constructor.construct_and_solve_hamiltonians()
    H_constructor.plot_figures()

