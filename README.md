# Intro
Some scripts for visualizing energy modes in 1d systems, written as supplement as I followed along with the [Topocondmat course](https://topocondmat.org/w1_topointro/1D.html).

# Scripts
### majorana_modes.py
Main script. Plot the energy levels for Kitaev or SSH chain. Parameters can also be tweaked via command line args.

### Examples:
```
# 1. Plot energies for Kitaev model as mu/t is tuned
python majorana_modes.py kitaev

# 2. as last-to-first-site hopping is tuned (default mu/t=0)
python majorana_modes.py kitaev --tune x 

# 3. with mu/t set at 3
py majorana_modes.py kitaev --tune x -m 3.0

# 4. Plot energies for SSH model as t2/t1 is tuned
python majorana_modes.py ssh 
```
In SSH model case, default tuning parameter is `t2/t1`  (ratio of the two hopping strengths). But you can also specify `--tune x` to tune the strength of periodic hopping in SSH model.

#### Optional args:
- `-N` (default `25`): Set number of sites
- `-m` (default `1`): Set onsite-energy mu value (ignored if mu is being tuned)
- `-d` (default `1`): Set value of Cooper pairing
- `-t` (default `1`): Set value of next-neighbor hopping (ignored if ssh model)
- `-x` (default `0.5`): Set parameter tuning last-to-first-site hopping. 
- `-t2` (default `1`): Set value of double-bond hopping (single-bond is always 1)
- `--plot-band-idxs` (default `0 1`): plot spatial distribution of these eigenstates, indexed by number (e.g. `0 1 2` includes third-lowest state).
- `--enforce-even-sites` (default `False`): in SSH model, enforce an even number of C-sites in the system.

# Demo:
To demo the code, let's follow the Topocondmat course and produce our own plots as we go along.

First, plot the energy bands for the 1d Kitaev chain as the ratio (mu/t) of on-site energy to nearest-neighbor hopping is tuned
