from meep_exp.simulation import Simulation, fresnel
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


C = 299_792_458


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


def arrange_medium(mask, width, height, thickness, a, epsilon):
    medium_list = []
    unit_cell = (width/a) / mask.shape[0]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y] == 1:
                material=mp.Medium(epsilon_diag=epsilon)
                medium = mp.Block(
                    size=mp.Vector3(unit_cell, unit_cell, thickness/a),
                    center=mp.Vector3(
                        unit_cell*x-(width/a)/2+unit_cell/2, 
                        (height/a)/2-unit_cell*y-unit_cell/2, 0
                    ),
                    material=material
                )
                medium_list.append(medium)

    return medium_list


a = 0.5e-6
resolution = 1/18
sx = sy = 270e-6
sz = 1500e-6
field_size = (sx, sy, sz)
f_center = 0.7e12
f_width = 1e12
source_center = -400e-6
pml_width = 120e-6
monitor_position = 500e-6
medium_width = 100e-6

mask = np.zeros((32, 32), dtype=np.int8)
unit = sx / len(mask)
mask[
    8:-8,
    8:-8
] = 1
# mask = load_mask('0.png', (sx, sy, medium_width)) # define metal-mesh pattern from image.

medium0 = arrange_medium(
    mask, sx, sy, medium_width, 
    a, mp.Vector3(1.69**2, 1.54**2, 1.54**2)
)
medium90 = arrange_medium(
    np.rot90(mask), sx, sy, medium_width, 
    a, mp.Vector3(1.54**2, 1.69**2, 1.54**2)
)

medium = {
    0: [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, medium_width/a),
            center=mp.Vector3(0,0,0),
            material=mp.perfect_electric_conductor
        ),
        mp.Block(
            size=mp.Vector3(135e-6/a, 135e-6/a, medium_width/a),
            center=mp.Vector3(0,0,0),
            material=mp.Medium(
                epsilon_diag=mp.Vector3(1.69**2, 1.54**2, 1.54**2)
            )
        )
    ],
    90: [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, medium_width/a),
            center=mp.Vector3(0,0,0),
            material=mp.perfect_electric_conductor
        ),
        *medium0
    ],
}

simulator = Simulation(
    f_center, f_width, source_center, 
    field_size, resolution, pml_width,
    medium, medium_width, monitor_position,
    sim_time=80e-12, n_freq=300, a=a
)
simulator.run()

simulator.save_spectrm('images/test', (0.5, 1.0))

simulator.save_data('images/test')