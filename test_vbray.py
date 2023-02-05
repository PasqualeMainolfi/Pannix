from ray_intersection import Ray
import numpy as np

r = Ray()

phi = 25 * np.pi/180

source = np.array([np.cos(phi), np.sin(phi)])
r.set_source_position(pos=source)

ls_phi = 45 * np.pi/180
ls1 = np.array([np.cos(-ls_phi), np.sin(-ls_phi)])
ls2 = np.array([np.cos(ls_phi), np.sin(ls_phi)])

ls = np.array([ls1, ls2]).T

cast = r.get_intersection(segment=ls)

r.show_intersection(source=source, segment=ls)