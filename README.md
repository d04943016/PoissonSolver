# PoissonSolver
A Poisson Solver for Semiconductor

--------
Author : Wei-Kai Lee
(Note) 2024/06/05 1D Poisson solver

-------

### Usage
1. Download [vscode](https://code.visualstudio.com/)
2. Open vscode
3. Open terminal
4. Install package
```
pip install numpy scipy matplotlib tqdm
```

### Example
In notebook,
* Poisson solver:
  * [Si PN homojunction](notebook/Poisson1D_cases/1.%20Si%20PN%20homojunction.ipynb)
  * [Two junctions](notebook/Poisson1D_cases/2.%20Two%20Junctions.ipynb)
  * [GaAs and Si heterojunction](notebook/Poisson1D_cases/3.%20GaAs%20and%20Si%20heterojunction.ipynb)
  * [Boundary condition/metal semiconductor junction](notebook/Poisson1D_cases/4.%20Boundary%20Condition%20(Metal%20Si%20junction).ipynb)
  * [Multiple Quantum Well (w/o considering quantum effect)](notebook/Poisson1D_cases/5.%20Multiple%20Quantum%20Well.ipynb)
*  Schrodinger solver:
   *  [Infinite Quantum Well](notebook/Schrodinger%201D_cases/1.%20Infinite%20Quantum%20Well.ipynb)
   *  [Simple Harmonic Oscillator / SHO](notebook/Schrodinger%201D_cases/2.%20SHO.ipynb)
   *  [Rectangular Quantum Well](notebook/Schrodinger%201D_cases/3.%20Rectangular%20Quantum%20Well.ipynb)
   *  [Triangular Quantum Well](notebook/Schrodinger%201D_cases/4.%20Trianglular%20Quantum%20Well.ipynb)
   *  [1/r Quantum Well](notebook/Schrodinger%201D_cases/5.%20Inverse%20r%20Quantum%20Well.ipynb)