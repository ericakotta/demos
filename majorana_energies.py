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
import cmath

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
        self.basis = kwargs.get('basis', 'single-particle')


        if self.model.lower() == 'kitaev':
            # Kitaev model is 1d chain of superconducing qubits characterized by parameters t, m, d
            # that specify nn-hopping strength, on-site energy, cooper pairing strength, respectively
            self.t = model_params.get('nn_hopping', 1.0)
            self.m = model_params.get('onsite_energy', 1.0)
            self.d = model_params.get('cooper_pairing', 1.0)

            self.descr = f"Kitaev t: {self.t}, d: {self.d}"
            if 'mu' not in self.param_label:
                self.descr += f", mu: {self.m}"
            
            if model_params['tuning_parameter'] == 'x':
                self.ROUTINE = 1
                print(f"Constructing BdG Hamiltoninan for Kitaev chain tuning N-to-1 hopping...")
            else:
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
                    self.N_sites, param*self.t, self.t, self.d,
                )
            elif self.ROUTINE == 1:
                H_BdG = construct_hamiltonian_kitaev(
                    self.N_sites, self.m, self.t, self.d, 
                    last_first_phase=1.*param,
                )
            elif self.ROUTINE == 2:
                H_BdG = construct_hamiltonian_ssh(
                    N, self.t1, param*self.t1,
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
            
            if param == self.param_space[0]:
                _, axs = plt.subplots(1,2, figsize=(10,4))
                sns.heatmap(ax=axs[0], data=np.real(H_BdG), annot=True, square=True)
                axs[0].set_title('real')
                sns.heatmap(ax=axs[1], data=np.imag(H_BdG), annot=True, square=True)
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
            plt.title(f'State density at {self.param_label}={round(self.param_space[param_idx],1)}')
            plt.xlabel('Atomic site number')
            for i, state_densities in enumerate(self.plot_state_densities[param_idx]):
                plt.plot(state_densities, alpha=0.5, label=f'{i}')
            if subplot_idx in subplot_idxs[:-1]:
                plt.gca().set_xticklabels([])
                plt.gca().set_xlabel('')
        
        plt.suptitle(self.descr)
        plt.legend(fontsize=8)
        plt.gca().legend(loc='upper center', bbox_to_anchor=(1.2, 1.05))
        plt.tight_layout()
        plt.show()


def construct_hamiltonian_kitaev(
        N, mu_onsite, t_nn, d_cooper,
        last_first_phase=0.5, apply_cooper_phase=True,
        antisymmetrize=False,
        basis='bogoliubov',
    ):
    '''
    N is number of atomic sites
    mu_onsite is onsite energy, t_nn is nearest-neighbor hopping, d_cooper is Cooper pairing strength

    boundary_hopping sets the strength of hopping t* between first and last site (closing the loop).
    The value of boundary_hopping is the x in t* = t_nn(1 - 2x) (knob to vary t* between -t_nn and t_nn)
    '''
    if basis=='majorana':
        ham_size = 2 * N
        H = np.zeros((ham_size,) * 2, dtype=np.complex128)
        for i in range(0, ham_size-3, 2):
            H[i, i+1] += 2. * 1j * mu_onsite
            H[i, i+3] += 2. * 1j * (d_cooper + t_nn)
            H[i+1, i+2] += 2. * 1j * (d_cooper - t_nn)
        H[-2, 1] += 2. * 1j * (d_cooper + t_nn) * np.cos(last_first_phase * np.pi )
        H[-1, 0] += 2. * 1j * (d_cooper - t_nn) * np.cos(last_first_phase * np.pi )
        H = 0.5 * (H - H.T)
        return H

    elif basis=='bogoliubov':
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
        # Add last-to-first hopping with phas
        Ht[0, N-1] += t_nn * (2 * last_first_phase - 1)  #np.cos(last_first_phase * np.pi )
        Ht[N-1, 0] += t_nn * (2 * last_first_phase - 1)  # np.cos(last_first_phase * np.pi )
        Ht[N, -1] += -t_nn * (2 * last_first_phase - 1)   # np.cos(last_first_phase * np.pi )
        Ht[-1, N] += -t_nn * (2 * last_first_phase - 1)  # np.cos(last_first_phase * np.pi )

        # Ht[0, -1] += -t_nn * np.real(cmath.exp(1j * last_first_phase * np.pi ))
        # Ht[N-1, N] += t_nn 
        # Ht[N-1, 0] += -t_nn * np.real(cmath.exp(1j * last_first_phase * np.pi ))
        # Ht[2*N-1, N] += t_nn * np.real(cmath.exp(1j * last_first_phase * np.pi ))
        # Ht = Ht + np.transpose(np.conjugate(Ht))

        # Get cooper-pairing contribution
        Hd = np.zeros((2*N,)*2, dtype=np.complex128)
        for i in range(N -1):
            Hd[i, N+i+1] += d_cooper
            Hd[i+1, N+i] += -d_cooper
            Hd[i+N, i+1] += -d_cooper
            Hd[i+N+1, i] += d_cooper

        if apply_cooper_phase:
            # Allow Cooper pairing between last and first site of chain
            Hd[N, N-1] += -d_cooper * (2 * last_first_phase - 1)  # np.real(cmath.exp(1j * last_first_phase * np.pi ))
            Hd[-1, 0] += d_cooper * (2 * last_first_phase - 1)  # np.real(cmath.exp(1j * last_first_phase * np.pi ))
            Hd[0, -1] += d_cooper * (2 * last_first_phase - 1)   # np.real(cmath.exp(1j * last_first_phase * np.pi ))
            Hd[N-1, N] += -d_cooper * (2 * last_first_phase - 1)  #np.real(cmath.exp(1j * last_first_phase * np.pi ))
            # Hd[N-1, N] += -d_cooper * np.real(np.exp(1j * last_first_phase * np.pi / 180))
            # Hd[2*N-1, 0] += d_cooper * np.real(np.exp(1j * last_first_phase * np.pi / 180))
            pass
        # Hd = Hd + np.transpose(np.conjugate(Hd))

        H = Hm + Ht + Hd

        if antisymmetrize:
            return antisymmetrize_H(H)
        else:
            return H

    


def construct_hamiltonian_kitaev_arx(N, mu_onsite, t_nn, d_cooper, last_first_hop=None):
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
        H_block[-1, 0] += t_nn * np.real(cmath.exp(1j * last_first_hop * np.pi ))
        H_block[0, -1] += t_nn *np.real(cmath.exp(1j * last_first_hop * np.pi )).conjugate()


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

def antisymmetrize_H(H_BdG):
    sz = int(H_BdG.shape[0] / 2)
    H = H_BdG[:sz, :sz]
    D = H_BdG[:sz, sz:]
    H_out = np.zeros(H_BdG.shape, dtype=np.cdouble)
    H_out[:sz, :sz] = H - H.conjugate() + D - D.conjugate()
    H_out[:sz, sz:] = 1j * (-H - H.conjugate() + D + D.conjugate())
    H_out[sz:, :sz] = 1j * (H + H.conjugate() + D + D.conjugate())
    H_out[sz:, sz:] = H - H.conjugate() - D + D.conjugate()
    return H_out

def construct_hamiltonian_ssh(
        N, t1, t2, 
        close_loop_hopping=None,
        even_sites=False,
        antisymmetrize=False
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

    N = args.N # Number of unit cells
    param_label = args.tune # Parameter to tune
    plot_band_idx = args.plot_band_idxs # Idx of energy modes to see spatial distribution
    # param_slices = args.param_values # Parameter values at which to plot spatial distribution

    if param_label == 'x':
        param_space = np.linspace(0., 1., 41)

    # Get model-defining parameters
    if args.model.lower() == 'kitaev':
        # Set up parameter space to tune system through
        if param_label is None or param_label == 'mu/t':
            param_label = 'mu/t'
            param_space = np.linspace(0.0, 3.0, 41)

        model_params = {
            'onsite_energy': parse_complex_arg(args.m), # Unused if tuning mu/t
            'cooper_pairing': parse_complex_arg(args.d), # Cooper pairing strength
            'nn_hopping': parse_complex_arg(args.t), # nearest-neighbor hopping
            'N_sites': N,
            'parameter_space': param_space,
            'tuning_parameter': param_label,
        }

    elif args.model.lower() == 'ssh':
        # Set up param space to tune system through
        if param_label is None:
            param_label = 't2/t1'
            param_space = np.linspace(0, 1.0, 41)
        
        model_params = {
            't1_hopping': 1.0 + 0.0j , # Reference strength for SSH single-bond hopping
            't2_hopping': parse_complex_arg(args.t2), # Not used if t2 is tuning parameter
            'N_sites': 2 * N,
            'parameter_space': param_space,
            'tuning_parameter': param_label,
        }

    # Initiate Hamiltonian constructor class
    H_constructor = H_BdG_constructor(
        args.model,
        model_params,
        enforce_even_sites=args.enforce_even_sites,
        plot_band_idxs=args.plot_band_idxs,
    )
    H_constructor.construct_and_solve_hamiltonians()
    H_constructor.plot_figures()

