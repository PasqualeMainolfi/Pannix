import panner as p
import numpy as np


LS = [135, 45, -45, -135]

d = p.DBAP(speakers_loc=LS, rolloff=6, weights=1)
v = p.VBAP(speakers_loc=LS)


phi = 90 * np.pi/180
xs = .001 * np.cos(phi)
ys = .001 * np.sin(phi)

xref, yref = xs, ys


d_gains = d.calculate_gains(source=[xs, ys], ref=None, spatial_blur=0.1)
v_gains = v.calculate_gains(source=[xs, ys], normalize=False)


print(d_gains)
print(v_gains)