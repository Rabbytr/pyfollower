# Follower Crowd Simulation

Follower Crowd Simulation (`Follower`) is a project that simulates the movement of a crowd of comformists with self-organizing characteristic. In `Follower`, [ORCA](https://gamma.cs.unc.edu/ORCA/) is adapted as the basic collision avoidance model. This project is writen in C++ and built using CMake. `pyfollower` is the python interfaces for `Follower` with many easy-to-use APIs.

### Getting started

To get started with Follower Crowd Simulation, you'll need to have Python 3.6+ installed on your system. You can download Python from the [official website](https://www.python.org/), and Follower Crowd Simulation can be installed using pip:

```shell
pip install pyfollower
```

The installation process of this project requires Cmake and C++ compilation environment, so if you are a Windows user, it is a better choice to use a [compiled wheel file](https://pypi.org/project/pyfollower/#files)

```shell
pip install pyfollower-xxx.whl
```

After you have installed it, you can simply use this simulation Engine. As an example(Please make sure the numpy and matplotlib is installed):

```python
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
            prev = 1.3 * dx / dist
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
```



### Contributing

If you'd like to contribute to Follower Crowd Simulation, feel free to submit a pull request. Please make sure that your code follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide and that any new features are thoroughly tested.