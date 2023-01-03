"""
VBAP and DBAP implementation

reference:
    vbap -> V. Pulkki, Virtual Sound Source Positioning Using Vector Base Amplitude Panning
    dbap -> T. Lossius, P. Baltazar, DBAP Distance-Based Amplitude Panning
         -> Speaker Placement Agnosticism: Improving the Distance-Based Amplitude Panning Algorithm
"""

import numpy as np
from scipy.spatial import ConvexHull


class Panner:

    TO_RAD = lambda x: x * np.pi/180
    CHECK_DEG = lambda x: x - 360 if x > 180 else x

    def __init__(self, speakers_loc: list) -> None:

        """
        speakers_loc: list[tuple|float], list of speakers location must be [(r, phi), ...] or [phi, phi, (r, phi), ...]
        """

        _phi_deg = []
        _r = []
        for p in speakers_loc:

            if isinstance(p, tuple):
                mag = p[0]
                phi = p[1]
            else:
                mag = 1
                phi = p

            _r.append(mag)
            _phi_deg.append(phi)

        self.phi_deg = list(map(Panner.CHECK_DEG, _phi_deg))
        self.phi_rad = np.asarray(list(map(Panner.TO_RAD, self.phi_deg)), dtype=float)
        self.r = np.asarray(_r, dtype=float)
        self.speakers_pos = self.get_speakers_pos(r=self.r, phi=self.phi_rad)

    def get_speakers_pos(self, r: float, phi: float) -> list:
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        p = np.asarray([x, y], dtype=float)
        return p


class VBAP(Panner):

    def __init__(self, speakers_loc: list):
        super().__init__(speakers_loc)
        self.pairs = ConvexHull(self.speakers_pos.T).simplices
    
    def find_arc(self, source: list):

        """
        source_pos: list, cartesian coordinate of source [x, y]
        """

        for pair in self.pairs:
            base = self.speakers_pos[:, pair]
            g = self.calculate_gains(source=source, base=base)
            if np.min(g) >= 0:
                return pair

    def calculate_gains(self, source: list, base: list|None = None, normalize: bool = True):
        
        if base is None:
            arc = self.find_arc(source=source)
            base = self.speakers_pos[:, arc]
            g = np.zeros(self.speakers_pos.shape[1], dtype=float)
        else:
            arc = np.arange(len(source))
            g = np.zeros(len(source), dtype=float)
        
        gains = np.linalg.inv(base) @ source
        g[arc] = gains

        if normalize:
            norm = np.linalg.norm(g)
            norm = norm if norm > 0 else 1
            g /= norm
            g *= np.sqrt(2)/2
        
        return g
    

class DBAP(Panner):

    def __init__(self, speakers_loc: list, rolloff: float = 6, weights: list|float = 1.0) -> None:
        super().__init__(speakers_loc)
        self.a = rolloff/(20 * np.log10(2))
        self.w = weights
        self.center = np.mean(self.speakers_pos.T, axis=0)
    
    def calc_distance(self, pos1: list, pos2: list, r: float|None = None):
        spat_blur = r**2 if r is not None else 0
        return np.sqrt(np.sum((pos2 - pos1)**2) + spat_blur)
    
    def get_speakers_distance(self, source: list, r: float|None = None):
        
        ls_pos = self.speakers_pos.T
        d = []
        for pos in ls_pos:
            dist = self.calc_distance(pos1=pos, pos2=source, r=r)
            d.append(dist)
        distance = np.asarray(d, dtype=float)
        return distance
    
    def get_b(self, ls_distance: list[float], p: float, eta: float):

        """
        b_i = (u_i/u_m((1/p) - 1))^2 + 1
        u_i = (d_i - max(d))^2_normalized + e

        m = index of the median distaced loudspeaker from the virtual source
        max(d) = the loudspeaker furthest from the virtual source
        e = small value to avoid 0 gain in the most distant loudspeaker, 
        typically set to r/N
        """

        u = (ls_distance - ls_distance.max())
        u_norm = np.linalg.norm(u)
        u = u/u_norm if u_norm > 0 else u
        u = u**2 + eta

        um = np.median(ls_distance)

        b = (u/um * ((1/p) + 1))**2 + 1

        return b

    
    def get_p(self, source: list[float], ref: list[float], r: float|None = None):

        """
        The variable p is the distance from a reference point in the field to 
        the most distant speaker, max(d_s) = max{d_s1, ..., d_sN }, 
        divided by the distance between the reference and the virtual source, 
        d_rs, clipped to 1.
    
            | q = max(d_s)/d_rs, if q < 1
        p = |
            | 1, otherwise
        """

        ds = self.get_speakers_distance(source=ref, r=r)
        max_ds = ds.max()

        drs = self.calc_distance(pos1=ref, pos2=source, r=r)

        q = max_ds/drs
        value = q if q < 1 else 1

        return value
    
    def get_k(self, d: float, p: float, b: float):

        """
        from...

        k = 1/√sum((w_i)^2/(d_i)^2a)
        a = R/(20log10(2))
        d = √(x_i - x_s)^2 + (y_i - y_s)^2 + (r_s)^2

        to...

        k = (p^2a)/√sum((b_i)^2(w_i)^2/(d_i)^2a)

            | q = max(d_s)/d_rs, if q < 1
        p = |
            | 1, otherwise

        """

        k_num = p**(2 * self.a)
        k_den = np.sqrt(np.sum((b**2 * self.w**2)/d**(2 * self.a)))
        k = k_num/k_den
        return k
    
    def get_spatial_blur(self, ref: list):

        """
        r = (sum(d_ic)/N) + r_scalar 
        with 0.2 <= r_scalar <= 0.2
        """

        dist_cent = self.get_speakers_distance(source=ref)
        rs = (np.sum(dist_cent)/len(dist_cent)) + 0.2
        return rs
    
    def calculate_gains(self, source: list, ref: list|None = None, spatial_blur: float|None = None):

        """
        get loudspeakers gain factors
        v_i = kw_ib_i/(d_i)^a

        source_pos: list, cartesian coordinate of source position
        ref: list or None, if tuple specify cartesian coordinate of reference point ina  field. If None, ref point will be a geometric center
        spatial_blur: float or None, if None calculate default r (see get_spatial_blur) else r = spatial_blur
        """

        if ref is None:
            ref = self.center
        
        if spatial_blur is not None:
            spat_blur = spatial_blur
        else:
            spat_blur = self.get_spatial_blur(ref=ref)

        eta = spat_blur/self.speakers_pos.shape[1]

        d = self.get_speakers_distance(source=source, r=spat_blur)
        p = self.get_p(source=source, ref=ref, r=spat_blur)
        b = self.get_b(ls_distance=d, p=p, eta=eta)
        k = self.get_k(d=d, p=p, b=b)
        w = self.w
        a = self.a

        v = (k * w * b)/d**a
        return v


    
        
