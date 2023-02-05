import pannix as p
import numpy as np
import time


LS = [135, 45, -45, -135]

# d = p.DBAP(loudspeaker_loc=LS, rolloff=6, weights=1)
d = p.DBAP(loudspeaker_num=70, rolloff=12, weights=1)
v = p.VBAP(loudspeaker_num=30)


phi = 50 * np.pi/180
xs = 0.3 * np.cos(phi)
ys = 0.3 * np.sin(phi)

xref, yref = xs, ys


d_gains = d.calculate_gains(source=[xs, ys], ref=None, spatial_blur=0.1)

v_gains = v.calculate_gains(source=[xs, ys], normalize=False, mode="ray")


# print(f"DBAP g = {d_gains}\nVBAP g = {v_gains}")


v.display_panning(source=(xs, ys))

# print(v_gains)
