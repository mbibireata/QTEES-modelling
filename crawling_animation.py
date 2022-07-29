import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from puppet2d import Puppet2d, pretty_range


"""
Set parameters.
"""

N_animals = 1                                                   # number of animals to simulate
channel_radius = 10                                             # inner radius of channel (segs)
channel_width = 1                                               # width of channel (segs)

duration = 500                                                  # duration of simulation
time_step = 0.1                                                 # time step for simulation
coordinate_std = 0.01                                            # standard deviation of initial position
momentum_std = 0.01                                              # standard deviation of initial momentum

fr = 10                                                         # animation frame rate decimation factor
SAVE_ANIMATION = True                                          # should we save the animation? (slow)
OUTPUT_FILENAME = "./channel_crawler_animation.mp4"             # where to save animation


"""
Construct N model larvae and run simulations.
"""

puppies = [Puppet2d(N=11, channel_radius=channel_radius, channel_width=channel_width) for i in range(N_animals)]

for i in pretty_range(N_animals) :
    print("Generating trajectory " + str(i + 1) + " of " + str(N_animals))
    puppy = puppies[i]
    initial_conditions = puppy.generate_initial_state(coordinate_std, momentum_std)
    time_axis, trajectory = puppy.generate_trajectory(initial_conditions, 1000, 0.1)


"""
Animate simulation outputs!
"""

# animated output
print("Animating simulation output...")

markersize = 2*(1 - np.cos(2*np.pi*puppies[0].space_axis) + 0.1)     # width of larva for animation

fig, axes = plt.subplots(1, num="channel crawler animation", figsize=(3, 3))
camera = Camera(fig)
    
plt.ylabel("$y$ (body lengths)")
plt.xlabel("$x$ (body lengths)")
frame_indices = np.arange(len(time_axis))[::fr]
    
for i in pretty_range(len(frame_indices)) :
    for puppy in puppies : 
        r = puppy.r[frame_indices[i]]
        rx = r[:, 0]
        ry = r[:, 1]
        plt.scatter(rx, ry, lw=2, marker="o", s=markersize, alpha=0.9, c='k')
        plt.plot(rx, ry, lw=1, alpha=0.9, c='k')
#        puppy.plot_configuration(puppy.trajectory[frame_indices[i]])
    camera.snap()
         
lim = puppy.outer_radius*1.5
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
    
#plt.axhline(puppies[0].channel_width/2, c='DarkGrey', zorder=-10)
#plt.axhline(-puppies[0].channel_width/2, c='DarkGrey', zorder=-10)

angles = np.linspace(0, 2*np.pi, 1000)
plt.plot(puppies[0].inner_radius*np.cos(angles),
         puppies[0].inner_radius*np.sin(angles), c='DarkGrey', zorder=-10)
plt.plot(puppies[0].outer_radius*np.cos(angles),
         puppies[0].outer_radius*np.sin(angles), c='DarkGrey', zorder=-10)
    
animation = camera.animate(interval=30)
    
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
    
if SAVE_ANIMATION :
    print("saving animation...")
    animation.save(OUTPUT_FILENAME, writer="imagemagick")
