# Gas Simulation in 2D 

## Overview

This project simulates a 2D ideal gas composed of elastic particles confined in a square box.
Each particle follows Newtonian dynamics with elastic collisions both between particles and with the box walls.
The simulation visualizes in real time:

* The **motion** of gas particles.
* The **velocity distribution**, compared with the **theoretical Maxwell-Boltzmann distribution**.
* The **kinetic energy distribution**, compared with the **Boltzmann-Gibbs law**.

It uses **Numba** for just-in-time compilation and parallelization to optimize computation speed.

---

## Features

1. 2D molecular dynamics with **elastic collisions**.
2. **Numba-accelerated** computation for real-time performance with thousands of particles.
3. Visualization of:

   * Particle motion with velocity-dependent coloring.
   * Velocity distribution histogram and its theoretical Maxwell-Boltzmann curve.
   * Kinetic energy distribution histogram and its theoretical Boltzmann curve.
4. Continuous updates of **instantaneous temperature** from mean kinetic energy.
5. Modular and extensible code for physics or visualization modifications.

---

## Dependencies

* **Python ≥ 3.9**
* **NumPy** (numerical arrays)
* **Matplotlib** (visualization and animation)
* **Numba** (JIT compilation and parallelization)

Install them using:

```bash
pip install numpy matplotlib numba
```

---

## File Structure

```
.
├── gas_simulation.py       # Main simulation script
└── README.md               # Documentation
```

---

## How to Run

Execute directly from the command line:

```bash
python ideal_gas_simulation.py
```

During execution:

* The terminal prints initialization information (parameters, optimization mode).
* A Matplotlib window opens showing:

  * Left: real-time particle motion in a box.
  * Center: velocity distribution histogram vs. theoretical Maxwell-Boltzmann curve.
  * Right: kinetic energy histogram vs. theoretical Boltzmann-Gibbs curve.

The simulation runs continuously until the window is closed.

---

## Physical Model

### 1. Motion

Each particle $i$ has:

* Position $\displaystyle \vec{r}_i = (x_i, y_i)$
* Velocity $\displaystyle \vec{v}_i = (v_{x,i}, v_{y,i})$

Position update (Euler scheme):
$$
\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \vec{v}_i \Delta t
$$

Boundary conditions:
Perfectly elastic collisions with box walls.

---

### 2. Inter-particle Collisions

Pairwise collisions are elastic and conserve momentum and kinetic energy.
The algorithm checks overlap between particles and adjusts positions and velocities accordingly using the normal component of relative velocity.

---

### 3. Distributions

#### Velocity (2D Maxwell-Boltzmann)

For a 2D gas, speed $v = \sqrt{v_x^2 + v_y^2}$ follows a **Rayleigh distribution**:

$$
f(v) = \frac{v}{\sigma^2} e^{-v^2 / (2\sigma^2)}, \quad \sigma^2 = \frac{k_B T}{m}
$$

#### Energy (Boltzmann-Gibbs)

For kinetic energy $E = \frac{1}{2} m v^2$:

$$
f(E) = \frac{1}{k_B T} e^{-E / (k_B T)}
$$

---

## Code Architecture

### Main Components

#### 1. Initialization

```python
initialize_particles(n, box_size, temp, mass)
```

Generates random initial positions and Maxwell-Boltzmann distributed velocities.

#### 2. Dynamics

* `update_positions`: integrates motion and handles wall collisions.
* `handle_collisions`: resolves inter-particle collisions.
* `compute_speeds`: computes speed magnitudes.
* `compute_kinetic_energies`: computes individual kinetic energies.

#### 3. Theoretical Models

* `maxwell_boltzmann_2D(v, T, m)`
* `energy_distribution(E, T)`

#### 4. Visualization

The `GasSimulation` class creates and updates:

* The particle scatter plot (colored by velocity magnitude).
* Velocity and energy histograms with theoretical curves.

Animation uses:

```python
FuncAnimation(sim.fig, sim.update, interval=1, blit=False, cache_frame_data=False)
```

---

## Adjustable Parameters

All parameters are defined at the beginning of the file:

| Parameter     | Description                          | Default |
| ------------- | ------------------------------------ | ------- |
| `N_PARTICLES` | Number of gas particles              | 2000    |
| `BOX_SIZE`    | Size of the simulation box           | 10.0    |
| `DT`          | Time step                            | 0.001   |
| `MASS`        | Particle mass                        | 1.0     |
| `RADIUS`      | Particle radius                      | 0.05    |
| `K_B`         | Boltzmann constant (arbitrary units) | 1.0     |
| `TEMP_INIT`   | Initial temperature                  | 100.0   |
| `N_BINS`      | Number of histogram bins             | 40      |

Modify these constants to change resolution, density, or performance.

---

## Performance Considerations

* The **Numba JIT** compiler drastically improves performance, particularly for large numbers of particles (`N_PARTICLES > 1000`).
* Parallel loops (`prange`) exploit multi-core CPUs.
* The complexity scales roughly as $O(N^2)$ due to pairwise collision checks. For much larger systems, **cell-based collision detection** would be required.

---

## Limitations

* Collisions are **simplified** and assume equal masses and hard-sphere behavior.
* No long-range forces or potential interactions.
* Time integration uses a simple **Euler scheme**, which is sufficient for visualization but not for precise physical modeling.
* Computation cost increases quadratically with particle number.

---

## Possible Extensions

1. Implement **Verlet integration** for better numerical stability.
2. Add **cell-linked lists** or **Verlet neighbor lists** to reduce collision complexity.
3. Introduce **external potentials** (e.g., gravity, harmonic trap).
4. Compute **macroscopic observables** (pressure, temperature fluctuations).
5. Extend to **3D** visualization.
6. Export data for statistical analysis (velocity, energy, autocorrelations).

---

## Validation

The simulation validates the **Maxwell-Boltzmann hypothesis**:
After thermalization, the velocity histogram converges to the theoretical Rayleigh distribution, and the mean kinetic energy matches $\langle E \rangle = k_B T$.

<img width="1599" height="572" alt="image" src="https://github.com/user-attachments/assets/090cf0e1-cf41-4987-90c5-83529be9bc18" />


---

## License

This project is open for academic and educational use.
You may modify and redistribute it with attribution to the original author.

---

## Author

Developed for scientific visualization and numerical experimentation in computational physics.
Designed for educational use to illustrate thermodynamic equilibrium in a 2D gas system.
