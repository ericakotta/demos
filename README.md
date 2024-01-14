# Intro
Some scripts for visualizing energy modes in 1d systems, written as supplement as I followed along with the [Topocondmat course](https://topocondmat.org/w1_topointro/1D.html).

## majorana_modes.py
Plot the energy levels for Kitaev or SSH chain. Parameters can also be tweaked via command line args.

### Examples:
```
# Plot energies for Kitaev model as mu/t is tuned:
py majorana_modes.py kitaev

# above but as last-to-first-site hopping is tuned (default mu/t=0)
py majorana_modes.py kitaev --tune x 

# above but with mu/t set to 3.0
py majorana_modes.py kitaev --tune x -m 3.0 
```
To do the same for the SSH model, replace `'kitaev'` arg with `'ssh'`. In SSH model case, default tuning parameter is ratio of the two hopping strengths `t2/t1`.

#### Optional args:
- `-N` (default `25`): Set number of sites
- `-m` (default `1`): Set onsite-energy mu value (ignored if mu is being tuned)
- `-d` (default `1`): Set value of Cooper pairing
- `-t` (default `1`): Set value of next-neighbor hopping (ignored if ssh model)
- `-x` (default `0.5`): Set parameter tuning last-to-first-site hopping. 
- `-t2` (default `1`): Set value of double-bond hopping (single-bond is always 1)
- `--plot-band-idxs` (default `0 1`): plot spatial distribution of these eigenstates, indexed by number (e.g. `0 1 2` includes third-lowest state).
- `--enforce-even-sites` (default `False`): in SSH model, enforce an even number of C-sites in the system.

