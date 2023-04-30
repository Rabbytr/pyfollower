import numpy as np
from pyfollower import FollowerEngine

# Initial scenario
dest = dict()
N = 50
sim = FollowerEngine(agent_radius=0.5)

for idx, i in enumerate(np.linspace(0, 1, N + 1)):
    if i == 1: break
    theta = i * np.pi * 2
    ox, oy = np.cos(theta), np.sin(theta)
    dx, dy = np.cos(theta + np.pi), np.sin(theta + np.pi)
    t = np.array([ox, oy, dx, dy]) * 20
    agent_id = sim.add_agent(*t)
    dest[agent_id] = t[-2:]

# Test obstacles
sim.add_obstacles([(5, 5), (-5, 5), (-5, -5), (5, -5)])
sim.process_obstacles()
obs = np.array([(-5, -5), (-5, 5), (5, 5), (5, -5), (-5, -5)])


# Run simulation --- Main loop
traj = []
for i in range(100):
    if not i % 10: print(sim.time)
    x = sim.get_agent_positions()
    for agent_id in range(N):
        dx = dest[agent_id] - x[agent_id, :]
        dist = np.sqrt(np.sum(dx ** 2))
        if dist < 0.5:
            prev = (0, 0)
        else:
            prev = 1.0 * dx / dist
        sim.set_agent_pref(agent_id, *prev)
    traj.append(x)
    print(x.T)
    sim.follower_step()


# Plot
import pylab as pl

traj = np.stack(traj)

pl.xlim(-20, 20)
pl.ylim(-20, 20)
pl.imshow(np.zeros((40, 40), float), extent=(-20, 20, -20, 20))

import matplotlib.cm as cm

colors = cm.hsv(np.linspace(0, 1, N))
for i in range(N):
    pl.plot(traj[:, i, 0], traj[:, i, 1], c=colors[i])

pl.scatter(*x.T, c=range(N), s=20, cmap='hsv')

pl.plot(obs[:, 0], obs[:, 1], c='white')
pl.show()