import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import Normalize
from numba import njit, prange
import time

# ==================== PARAMETERS ====================
N_PARTICLES = 2000   # Number of particles
BOX_SIZE = 10.0      # Size of the box
DT = 0.001           # Time step
MASS = 1.0           # Particle mass
RADIUS = 0.05        # Particle radius
K_B = 1.0            # Boltzmann constant (arbitrary units)
TEMP_INIT = 100.0    # Initial temperature
N_BINS = 40          # Number of bins for histograms

# ==================== NUMBA-OPTIMIZED FUNCTIONS ====================

@njit(parallel=True, fastmath=True)
def initialize_particles(n, box_size, temp, mass):
    """Initialize particle positions and velocities"""
    positions = np.random.uniform(0, box_size, (n, 2))
    
    # Maxwell-Boltzmann velocity distribution
    sigma = np.sqrt(K_B * temp / mass)
    velocities = np.random.normal(0, sigma, (n, 2))
    
    return positions, velocities

@njit(parallel=True, fastmath=True)
def update_positions(positions, velocities, dt, box_size):
    """Update positions with wall collision handling"""
    n = positions.shape[0]
    
    for i in prange(n):
        for j in range(2):
            positions[i, j] += velocities[i, j] * dt
            
            # Elastic collision with walls
            if positions[i, j] < 0:
                positions[i, j] = -positions[i, j]
                velocities[i, j] = -velocities[i, j]
            elif positions[i, j] > box_size:
                positions[i, j] = 2 * box_size - positions[i, j]
                velocities[i, j] = -velocities[i, j]
    
    return positions, velocities

@njit(parallel=True, fastmath=True)
def handle_collisions(positions, velocities, radius):
    """Simplified pairwise particle collision handling"""
    n = positions.shape[0]
    
    for i in prange(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = 2 * radius
            
            if dist_sq < min_dist * min_dist and dist_sq > 0:
                dist = np.sqrt(dist_sq)
                
                # Normal vector
                nx = dx / dist
                ny = dy / dist
                
                # Relative velocities
                dvx = velocities[j, 0] - velocities[i, 0]
                dvy = velocities[j, 1] - velocities[i, 1]
                
                # Projection onto normal
                dvn = dvx * nx + dvy * ny
                
                # Elastic collision
                if dvn < 0:
                    velocities[i, 0] += dvn * nx
                    velocities[i, 1] += dvn * ny
                    velocities[j, 0] -= dvn * nx
                    velocities[j, 1] -= dvn * ny
                    
                    # Separate overlapping particles
                    overlap = min_dist - dist
                    positions[i, 0] -= overlap * 0.5 * nx
                    positions[i, 1] -= overlap * 0.5 * ny
                    positions[j, 0] += overlap * 0.5 * nx
                    positions[j, 1] += overlap * 0.5 * ny
    
    return positions, velocities

@njit(parallel=True, fastmath=True)
def compute_speeds(velocities):
    """Compute the magnitude of particle velocities"""
    n = velocities.shape[0]
    speeds = np.empty(n)
    
    for i in prange(n):
        speeds[i] = np.sqrt(velocities[i, 0]**2 + velocities[i, 1]**2)
    
    return speeds

@njit(parallel=True, fastmath=True)
def compute_kinetic_energies(velocities, mass):
    """Compute kinetic energy of each particle"""
    n = velocities.shape[0]
    energies = np.empty(n)
    
    for i in prange(n):
        v_sq = velocities[i, 0]**2 + velocities[i, 1]**2
        energies[i] = 0.5 * mass * v_sq
    
    return energies

# ==================== THEORETICAL DISTRIBUTIONS ====================

def maxwell_boltzmann_2D(v, temp, mass):
    """
    2D Maxwell-Boltzmann speed distribution (Rayleigh distribution).
    
    If vx, vy ~ N(0, σ²) with σ² = k_B*T/m,
    then v = sqrt(vx² + vy²) follows this distribution.
    """
    sigma_sq = K_B * temp / mass
    return (v / sigma_sq) * np.exp(-v**2 / (2 * sigma_sq))

def energy_distribution(E, temp):
    """Boltzmann distribution of kinetic energies"""
    return (1 / (K_B * temp)) * np.exp(-E / (K_B * temp))

# ==================== GAS SIMULATION CLASS ====================

class GasSimulation:
    def __init__(self, n_particles=N_PARTICLES, n_bins=N_BINS):
        self.n = n_particles
        self.n_bins = n_bins
        self.positions, self.velocities = initialize_particles(
            self.n, BOX_SIZE, TEMP_INIT, MASS
        )
        
        # === Figure configuration ===
        self.fig = plt.figure(figsize=(17, 6))
        gs = self.fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Subplot 1: particle motion
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        
        # Colormap (blue → red)
        self.cmap = cm.get_cmap('coolwarm')
        
        # RMS velocity (for color normalization)
        v_rms_theory = np.sqrt(2 * K_B * TEMP_INIT / MASS)
        self.v_min = 0
        self.v_max = v_rms_theory * 3
        
        # Colormap normalization
        self.norm = Normalize(vmin=self.v_min, vmax=self.v_max)
        
        # Initialize particle colors based on initial speeds
        speeds_init = compute_speeds(self.velocities)
        colors_init = self.cmap(self.norm(speeds_init))
        
        self.scatter = self.ax1.scatter(
            self.positions[:, 0], self.positions[:, 1], 
            s=20, c=colors_init, alpha=0.8, edgecolors='black', linewidths=0.3
        )
        
        self.ax1.set_xlim(0, BOX_SIZE)
        self.ax1.set_ylim(0, BOX_SIZE)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Gas Simulation', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Position X')
        self.ax1.set_ylabel('Position Y')
        
        # Add colorbar for speed
        sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax1, orientation='vertical', 
                                  pad=0.02, fraction=0.046)
        cbar.set_label('Velocity', rotation=270, labelpad=20, fontsize=11)
        
        # Subplot 2: velocity distribution
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title('Velocity Distribution (Maxwell-Boltzmann)', 
                          fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Velocity')
        self.ax2.set_ylabel('Probability Density')
        self.ax2.grid(True, alpha=0.3)
        
        # Subplot 3: energy distribution
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax3.set_title('Energy Distribution (Boltzmann-Gibbs)', 
                          fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Kinetic Energy')
        self.ax3.set_ylabel('Probability Density')
        self.ax3.grid(True, alpha=0.3)
        
        self.time = 0
        self.frame_count = 0
        
        # === Theoretical curves (computed once) ===
        v_rms = np.sqrt(2 * K_B * TEMP_INIT / MASS)
        self.v_theory = np.linspace(0, v_rms * 4, 300)
        self.maxwell_theory = maxwell_boltzmann_2D(self.v_theory, TEMP_INIT, MASS)
        
        E_mean_theory = K_B * TEMP_INIT
        self.E_theory = np.linspace(0, E_mean_theory * 5, 300)
        self.energy_dist_theory = energy_distribution(self.E_theory, TEMP_INIT)
        
        # Plot theoretical reference curves
        self.theory_line_velocity, = self.ax2.plot(
            self.v_theory, self.maxwell_theory, 'r-', linewidth=2.5,
            label=f'Theoretical 2D Maxwell-Boltzmann (T={TEMP_INIT:.2f})', 
            zorder=10
        )
        
        self.theory_line_energy, = self.ax3.plot(
            self.E_theory, self.energy_dist_theory, 'r-', linewidth=2.5,
            label=f'Theoretical Boltzmann (T={TEMP_INIT:.2f})', 
            zorder=10
        )
        
        # Axis configuration
        self.ax2.set_xlim(0, np.max(self.v_theory))
        self.ax2.legend(loc='upper right')
        
        self.ax3.set_xlim(0, np.max(self.E_theory))
        self.ax3.legend(loc='upper right')
        
        # Containers to hold histogram references
        self.hist_velocity_container = None
        self.hist_energy_container = None
        
    def update(self, frame):
        """Update one animation frame"""
        # Physics updates (multiple small steps for smooth motion)
        for _ in range(5):
            self.positions, self.velocities = update_positions(
                self.positions, self.velocities, DT, BOX_SIZE
            )
            self.positions, self.velocities = handle_collisions(
                self.positions, self.velocities, RADIUS
            )
            self.time += DT
        
        # Compute observables
        speeds = compute_speeds(self.velocities)
        energies = compute_kinetic_energies(self.velocities, MASS)
        
        # Instantaneous temperature from equipartition: <E> = k_B * T (2D)
        mean_energy = np.mean(energies)
        temp_instant = mean_energy / K_B
        
        # Update particle positions and colors
        self.scatter.set_offsets(self.positions)
        colors = self.cmap(self.norm(speeds))
        self.scatter.set_facecolors(colors)
        
        # ===== Velocity histogram =====
        if self.hist_velocity_container is not None:
            for patch in self.hist_velocity_container[2]:
                patch.remove()
        
        self.hist_velocity_container = self.ax2.hist(
            speeds, bins=self.n_bins, density=True, alpha=0.6, color='blue',
            label='Simulation Histogram', edgecolor='black', 
            linewidth=0.5, zorder=5
        )
        
        # ===== Energy histogram =====
        if self.hist_energy_container is not None:
            for patch in self.hist_energy_container[2]:
                patch.remove()
        
        self.hist_energy_container = self.ax3.hist(
            energies, bins=self.n_bins, density=True, alpha=0.6, color='green',
            label='Simulation Histogram', edgecolor='black', 
            linewidth=0.5, zorder=5
        )
        
        # Update title with statistics
        self.ax1.set_title(
            f'Gas Simulation\n'
            f'Particles: {self.n}| '
            f'Instantaneous T: {temp_instant:.3f} K',
            fontsize=11, fontweight='bold'
        )
        
        self.frame_count += 1
        
        return self.scatter,

# ==================== EXECUTION ====================

if __name__ == "__main__":
    print("🚀 Initializing gas simulation...")
    print(f"📊 Number of particles: {N_PARTICLES}")
    print(f"📊 Number of bins: {N_BINS}")
    print(f"⚡ Numba optimization with parallelization enabled")
    print(f"🔬 Initial temperature: {TEMP_INIT}")
    print(f"🎨 Color code: BLUE (slow) → RED (fast)")
    print("\n▶️  Starting simulation...\n")
    
    sim = GasSimulation(n_particles=N_PARTICLES, n_bins=N_BINS)
    
    # Animation
    anim = FuncAnimation(sim.fig, sim.update, interval=1, 
                        blit=False, cache_frame_data=False)
    
    plt.show()
    
    print("\n✅ Simulation finished!")
