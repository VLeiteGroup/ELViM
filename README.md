# ELViM: Energy Landscape Visualization Method

ELViM is method for visualizing the energy landscapes of biomolecules simulations.

## Requirements

### MDTraj
MDTraj is a python library that allows users to manipulate molecular dynamics (MD) trajectories. 
To install MDTraj using conda, execute the following command:

```bash
conda install -c conda-forge mdtraj
```
### Numba
Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.
To install Numba using conda, execute the following command:

```bash
conda install numba
```

## Execution
The ELViM code runs with flags to control all possible parameters. To see the options, use
python ELViM.py -h

## Example
To run an standard ELViM projection it is only necessary to have the trajectory file.

```bash
python ELViM.py -f trajectory.pdb -o output.dat
```
Alternatively, for a xtc file you do required a topology file.

```bash
python ELViM.py -f trajectory.xtc -t topology.pdb -o output.dat
```

## Citation
Please cite the following papers when using ELViM:

1. Oliveira Jr, A. B., Yang, H., Whitford, P. C., & Leite, V. B. (2019). Distinguishing biomolecular pathways and metastable states. Journal of chemical theory and computation, 15(11), 6482-6490. 
2. Viegas, R. G., Martins, I. B. S., Sanches M. N., Oliveira Jr, A. B., Camargo, J. B., Paulovich, F. V., & Leite, V. B. (2023). ELViM: Exploring Biomolecular Energy Landscapes through Multidimensional Visualization. 
