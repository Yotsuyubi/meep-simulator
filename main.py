from meep_exp.simulation import Simulation
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_mask(path, size):
    im = Image.open(path)
    im = im.resize((size[0], size[1]))
    im = im.convert('L')
    array = np.array(im) / 255
    array = (array < 1)
    array = np.abs(array - 1)
    array = np.expand_dims(array, -1)
    array = np.concatenate([array for _ in range(size[2])], -1)
    return array



resolution = 1/5
sx = sy = 270
sz = 1500
field_size = (sx, sy, sz)
f_center = 0.7e12
f_width = 1e12
source_center = -200
pml_width = 120
monitor_position = 400
medium_width = 10

mask = np.ones((sx, sy, medium_width), dtype=np.int8)
mask[70:-70,70:-70,:] = 0
# mask = load_mask('0.png', (sx, sy, medium_width)) # define metal-mesh pattern from image.

medium = {
    0: [
        mp.Block(
            size=mp.Vector3(sx, sy, medium_width),
            center=mp.Vector3(0,0,0),
            material=mp.MaterialGrid(
                grid_size=mp.Vector3(sx, sy, medium_width),
                medium1=mp.Medium(
                    epsilon_diag=mp.Vector3(1.69**2, 1.54**2, 1.54**2)
                ),
                medium2=mp.perfect_electric_conductor,
                weights=mask
            )
        )
    ],
    90: [
        mp.Block(
            size=mp.Vector3(sx, sy, medium_width),
            center=mp.Vector3(0,0,0),
            material=mp.MaterialGrid(
                grid_size=mp.Vector3(sx, sy, medium_width),
                medium1=mp.Medium(
                    epsilon_diag=mp.Vector3(1.54**2, 1.69**2, 1.54**2)
                ),
                medium2=mp.perfect_electric_conductor,
                weights=np.rot90(mask, axes=(0, 1), k=-1)
            )
        )
    ],
}

simulator = Simulation(
    f_center, f_width, source_center, 
    field_size, resolution, pml_width,
    medium, medium_width, monitor_position,
    sim_time=100e3, n_freq=300
)
# simulator.run(
#     tran_inc='tran_incident.npy', 
#     refl_inc='refl_incident.npy', 
#     refl_straight='refl_straight.npy'
# )
simulator.run()

simulator.save_incidents()

simulator.save_spectrm('images/test', (0.5, 1.0))

simulator.save_data('images/test')
