import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 500
m = 120*1.66E-27
T = 1000
kB = 1.380649e-23
v_0 = np.sqrt(3/2 * kB * T * 2 / m)
N_mean = 100
v_bins = np.arange(0, 2*v_0, 10)
dt = 0.00005

plt.rcParams["font.size"] = 14
np.random.seed(0)
particles = [[i, np.random.uniform([-1, -1], [1, 1], size=2), v_0 * np.array([np.cos(np.random.uniform(-1, 1) * 2 * np.pi), np.sin(np.random.uniform(-1, 1) * 2 * np.pi)]), 1E-2, m, "green"] for i in range(N)]

def collisions(particles, X, Y):
    new_list = []
    for p1 in particles:
        if p1 in new_list:
            continue
        x, y = p1[1]
        if ((x > X/2 - p1[3]) or (x < -X/2+p1[3])):
            p1[2][0] *= -1
        if ((y > Y/2 - p1[3]) or (y < -Y/2+p1[3])):
            p1[2][1] *= -1
        for p2 in particles:
            if p1 == p2:
                continue
            m1, m2, r1, r2, v1, v2 = p1[4], p2[4], p1[1], p2[1], p1[2], p2[2]
            if np.dot(r1-r2, r1-r2) <= (p1[3] + p2[3])**2:
                v1_new = v1 - 2*m1 / (m1+m2) * np.dot(v1-v2, r1-r2) / np.dot(r1-r2, r1-r2)*(r1-r2)
                v2_new = v2 - 2*m1 / (m1+m2) * np.dot(v2-v1, r2-r1) / np.dot(r2-r1, r2-r1)*(r2-r1)
                p1[2] = v1_new
                p2[2] = v2_new
                new_list.append(p2)

def motion(particles, dt):
    collisions(particles, 2, 2)
    for p in particles:
        p[1] += dt * p[2]

def v_p(particles):
    return [np.sqrt(np.dot(p[2], p[2])) for p in particles]


fig, (part_ax, dist_ax) = plt.subplots(figsize=(12, 6), ncols=2)
part_ax.set_xticks([])
part_ax.set_yticks([])
part_ax.set_xlim(-1, 1)
part_ax.set_ylim(-1, 1)
part_ax.set_aspect("equal")
part_ax.set_title("Gas Simulation")

dist_ax.set_xlim(0, 2*v_0)
dist_ax.set_ylim(0, 1.1*max(25*N*(m/(2*np.pi*kB*T))**(3/2) * 4 * np.pi*v_bins**2 * np.exp(-m*v_bins**2/(2*kB*T))))
dist_ax.set_xlabel("Particle Speed (m/s)")
dist_ax.set_ylabel("# of particles")
dist_ax.set_title("Distribution")

scatter = part_ax.scatter([], [])
histo = dist_ax.bar(v_bins, [0]*len(v_bins), width=0.9 * np.gradient(v_bins), align="edge", alpha=0.5, color='darkgreen')
dist_ax.plot(np.arange(0, 2*v_0, 1), 25*N*(m/(2*np.pi*kB*T))**(3/2) * 4 * np.pi*np.arange(0, 2*v_0, 1)**2 * np.exp(-m*np.arange(0, 2*v_0, 1)**2/(2*kB*T)), color="black")
T_txt = part_ax.text(-0.99, 0.9, s="")
freq_v = np.tile((np.histogram(v_p(particles), bins=v_bins)[0].astype(np.float)), (N_mean, 1))

def init():
    return scatter, *histo.patches

def update(frame):
    motion(particles, dt)
    temperature = np.mean(v_p(particles))**2 * m / (3 * kB)
    T_txt.set_text(f"T= {temperature:.2f} K")
    freqs, bins = np.histogram(v_p(particles), bins=v_bins)
    freq_v [frame % N_mean] = freqs
    freqs_mean = np.mean(freq_v , axis=0)
    for rect, height in zip(histo.patches, freqs_mean):
        rect.set_height(height)
    scatter.set_offsets(np.array([p[1] for p in particles]))
    scatter.set_color([p[5] for p in particles])   
    return scatter, *histo.patches, T_txt

ani = FuncAnimation(fig, update, frames=range(500), init_func=init, blit=True, interval=1, repeat=False)

plt.show()
