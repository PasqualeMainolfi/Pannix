
from panner import Panner
from scipy.spatial import ConvexHull
import numpy as np


class VBAP(Panner):

    def __init__(self, speakers_loc: list):
        super().__init__(speakers_loc)
        self.pairs = ConvexHull(self.speakers_pos.T).simplices
    
    def find_arc(self, source: list):

        """
        source_pos: list, cartesian coordinate of source [x, y]
        """
        
        x, y = source[0], source[1]
        phi = np.arctan2(y, x)

        if phi in self.phi_deg:
            arc = np.asarray(self.phi_deg.index(phi))
        else:
            arc = np.asarray([-1, -1], dtype=float)
            for pair in self.pairs:
                base = self.speakers_pos[:, pair]
                g = self.calculate_gains(source=source, base=base)
                if g.min() > 0:
                    arc = g
                    break
        return pair

    def calculate_gains(self, source: list, base: list|None = None, normalize: bool = True):
        
        if base is None:
            arc = self.find_arc(source=source)
            base = self.speakers_pos[:, arc]
            g = np.zeros(self.speakers_pos.shape[1], dtype=float)
        else:
            arc = np.arange(len(source))
            g = np.zeros(len(source), dtype=float)
        
        if base.ndim > 1:
            gains = np.linalg.inv(base) @ source
        else:
            gains = 1
        
        g[arc] = gains

        if normalize:
            norm = np.linalg.norm(g)
            norm = norm if norm > 0 else 1
            g /= norm
            g *= np.sqrt(2)/2
        
        return g