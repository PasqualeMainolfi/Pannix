"""
VBAP and DBAP implementation

reference:
    vbap -> V. Pulkki, Virtual Sound Source Positioning Using Vector Base Amplitude Panning
    dbap -> T. Lossius, P. Baltazar, DBAP Distance-Based Amplitude Panning
         -> J. Sundstrom, Speaker Placement Agnosticism: Improving the Distance-Based Amplitude Panning Algorithm  

moreover, the active arc searching function (2D) has been modified, implementing a line-line intersection algortithm
which allows for speeding up the computation (see ray_intersection.py).

reference: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection  

If you use this version of VBAP algo, please cite me.
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from ray_intersection import Ray


class Pannix:

    TO_RAD = lambda x: x * np.pi/180
    CHECK_DEG = lambda x: x - 360 if x > 180 else x

    def __init__(
    
        self, 
        loudspeaker_loc: list|str, 
        loudspeaker_num: int, 
    
    ) -> None:

        """
        loudspeaker_loc: list[tuple|float]|None, if list, pass a list of loudspeaker location ->
        must be [(r, phi), ...] or [phi, phi, (r, phi), ...]. If None, specify
        only the number of loudspeaker
        loudspeaker_num: int|None, number of loudspeaker if loudspeaker_loc is None
        """

        loc = self.__define_loudspeaker_loc(loc=loudspeaker_loc, num=loudspeaker_num)

        _phi_deg = []
        _r = []
        for p in loc:

            if isinstance(p, tuple):
                mag = p[0]
                phi = p[1]
            else:
                mag = 1
                phi = p

            _r.append(mag)
            _phi_deg.append(phi)

        self.phi_deg = list(map(Pannix.CHECK_DEG, _phi_deg))
        self.phi_rad = np.asarray(list(map(Pannix.TO_RAD, self.phi_deg)), dtype=float)
        self.r = np.asarray(_r, dtype=float)
        self.loudspeaker_pos = self.get_loudspeaker_pos(r=self.r, phi=self.phi_rad)
    
    def __define_loudspeaker_loc(self, loc: list|str, num: int) -> list:

        if isinstance(loc, list):
            return loc
        else:
            offset = 180/num
            step = 360/num
            s_loc = []
            for i in range(num):
                s_loc.append(step * i + offset)
            return s_loc
            
    def get_loudspeaker_pos(self, r: float, phi: float) -> list:

        """
        calculate loudspeaker position
        """
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        p = np.asarray([x, y], dtype=float)
        return p
    
    def car_to_pol(self, x: float, y: float) -> tuple:

        """
        from cartesian to polar

        x: float
        y: float

        return: tuple, (rho, phi)
        """

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return (rho, phi)
    
    def pol_to_cart(self, rho: float, phi: float, mode: str = "rad") -> tuple:

        """
        from polar to cartesian

        rho: float
        phi: float
        mode: str, rad or deg

        return: tuple, (x, y)
        """

        if mode == "deg":
            phi = Pannix.TO_RAD(Pannix.CHECK_DEG(phi))

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        return (x, y)
    
    def plot_panning(self, source: list|tuple, g: list|None, base:list, mode: str):

        """
        plot loudspeaker and source position

        source: tuple, source position (cartesian coordinate)
        g: list|None, if list plot loudspeaker gains. If None plot only loudspeaker layout
        mode: str, vbap or dbap
        """

        plt.style.use("ggplot")

        posx = self.loudspeaker_pos[0, :]
        posy = self.loudspeaker_pos[1, :]

        last_posx = [posx[-1], posx[0]]
        last_posy = [posy[-1], posy[0]]

        if g is None:
            plt.title("LOUDSPEAKER LAYOUT", weight="bold")
            plt.plot(posx, posy, "-o", c="k", lw=0.3)
            plt.plot(last_posx, last_posy, "-o", c="k", lw=0.3)
            plt.scatter(source[0], source[1], c="r", s=70)
            plt.plot([0, source[0]], [0, source[1]], c="r", lw=0.1)
            plt.annotate("S", source, ha="center", xytext=(-10, 0), textcoords="offset points")
            plt.xlabel("x")
            plt.ylabel("y")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))

            ax[0].set_title("LOUDSPEAKER LAYOUT", weight="bold")
            ax[0].plot(posx, posy, "-o", c="k", lw=0.3, zorder=1)
            ax[0].plot(last_posx, last_posy, "-o", c="k", lw=0.3, zorder=1)
            if mode == "dbap":
                for i in range(len(posx)):
                    ax[0].plot((source[0], posx[i]), (source[1], posy[i]), c="b", lw=0.5, zorder=1)
            else:
                for i in base:
                    ax[0].plot((source[0], posx[i]), (source[1], posy[i]), c="b", lw=0.5, zorder=1)
            ax[0].scatter(source[0], source[1], c="r", s=70)
            # ax[0].plot([0, source[0]], [0, source[1]], c="r", lw=0.1) # sorgente
            ax[0].annotate("S", source, ha="center", xytext=(-10, 0), textcoords="offset points", weight="bold")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")

            ax[1].set_title("GAINS", weight="bold")
            ax[1].bar([f"{x+1}" for x in range(len(g))], g, color="b", width=0.3)
            ax[1].set_xlabel("speakers")
            ax[1].set_ylabel("g")

        plt.subplots_adjust(wspace=0.3)
        plt.grid(alpha=0.7, zorder=0)
        plt.show()



class VBAP(Pannix):

    def __init__(

        self, 
        loudspeaker_loc: list|str = "auto", 
        loudspeaker_num: int = 4,

        ) -> None:

        """
        constructor

        loudspeaker_loc: list, loudspeaker location (see Pannix class)
        """

        super().__init__(loudspeaker_loc, loudspeaker_num)
        length = len(self.loudspeaker_pos[1])
        if length == 2:
            self.pairs = np.array([[0, 1]], dtype=int)
        elif length > 2:
            self.pairs = ConvexHull(self.loudspeaker_pos.T).simplices
        else:
            raise Exception("[ERROR] must be at least two loudspeakers!")
        
        self.__mode_find_arc = "default"
        
    
    def find_arc(self, source: list, angle: float) -> list:

        """
        find active arc.
        return index of active arc.

        source: list, cartesian coordinate of source [x, y]
        angle: float, pos angle in rad
        """

        if angle in self.phi_rad:
            print("ok")
            return np.where(angle==self.phi_rad)[0][0]
        else:
            for pair in self.pairs:
                base = self.loudspeaker_pos[:, pair]
                g = self.calculate_gains(source=source, base=base)
                if np.min(g) >= 0:
                    return pair
    
    def ray_cast_find_arc(self, ray: Ray, angle: float):

        if angle in self.phi_rad:
            return np.where(angle==self.phi_rad)[0][0]
        else:
            for pair in self.pairs:
                base = self.loudspeaker_pos[:, pair]
                cast = ray.get_intersection(segment=base)
                if cast is not None:
                    return pair
            return None


    def calculate_gains(self, source: list, base: list|None = None, normalize: bool = True, mode: str = "default") -> list:

        """
        calculate gains

        source: list, source position (cartesian)
        base: list, if you know the base (used in find_arc)
        normalize: bool, if True gains will be normalized
        mode: str, "default" = pulkki method, "ray" = ray cast find arc
        """

        self.__mode_find_arc = mode

        angle = np.arctan2(source[1], source[0])
        
        if base is None:
            if mode == "default":
                arc = self.find_arc(source=source, angle=angle)
            elif mode == "ray":
                r = Ray()
                r.set_position(pos=source)
                arc = self.ray_cast_find_arc(ray=r, angle=angle)
                arc = arc if arc is not None else 0
            else:
                print("[ERROR] mode not implemented... must be default or ray!")
                exit()
            base = self.loudspeaker_pos[:, arc]
            g = np.zeros(self.loudspeaker_pos.shape[1], dtype=float)
        else:
            arc = np.arange(len(source))
            g = np.zeros(len(source), dtype=float)
        
        if base.ndim > 1:
            gains = np.linalg.inv(base) @ source
        else:
            gains = 1 if mode == "default" else arc

        g[arc] = gains

        if normalize:
            norm = np.linalg.norm(g)
            norm = norm if norm > 0 else 1
            g /= norm
            g *= np.sqrt(2)/2
        
        return g

    def display_panning(self, source: list|tuple):
        g = self.calculate_gains(source=source, normalize=False, mode=self.__mode_find_arc)
        base = [i for i in range(len(g)) if g[i] != 0]
        self.plot_panning(source=source, g=g, base=base, mode="vbap")


class DBAP(Pannix):

    def __init__(

        self, loudspeaker_loc: list|str = "auto", 
        loudspeaker_num: int = 4, 
        rolloff: float = 6, 
        weights: list|float = 1.0

        ) -> None:
        
        """
        constructor

        loudspeaker_loc: list, loudspeaker location (see Pannix class)
        rolloff: float, rolloff coefficient
        weights: list|float, loudspeaker weights
        """
        
        super().__init__(loudspeaker_loc, loudspeaker_num)
        self.a = rolloff/(20 * np.log10(2))
        self.w = weights
        self.center = np.mean(self.loudspeaker_pos.T, axis=0)
    
    def calc_distance(self, pos1: list, pos2: list, r: float|None = None):

        """
        calculate distance
        """

        spat_blur = r**2 if r is not None else 0
        return np.sqrt(np.sum((pos2 - pos1)**2) + spat_blur)
    
    def get_loudspeaker_distance(self, source: list, r: float|None = None) -> list[float]:

        """
        calculate distance between loudspeaker and source
        """
        
        ls_pos = self.loudspeaker_pos.T
        d = []
        for pos in ls_pos:
            dist = self.calc_distance(pos1=pos, pos2=source, r=r)
            d.append(dist)
        distance = np.asarray(d, dtype=float)
        return distance
    
    def get_b(self, ls_distance: list[float], p: float, eta: float) -> list[float]:

        """
        b_i = (u_i/u_m((1/p) - 1))^2 + 1
        u_i = (d_i - max(d))^2_normalized + e

        m = index of the median distaced loudloudspeaker from the virtual source
        max(d) = the loudloudspeaker furthest from the virtual source
        e = small value to avoid 0 gain in the most distant loudloudspeaker, 
        typically set to r/N
        """

        u = (ls_distance - ls_distance.max())
        u_norm = np.linalg.norm(u)
        u = u/u_norm if u_norm > 0 else u
        u = u**2 + eta

        um = np.median(ls_distance)

        b = (u/um * ((1/p) + 1))**2 + 1

        return b

    
    def get_p(self, source: list[float], ref: list[float], r: float|None = None) -> float:

        """
        The variable p is the distance from a reference point in the field to 
        the most distant loudspeaker, max(d_s) = max{d_s1, ..., d_sN }, 
        divided by the distance between the reference and the virtual source, 
        d_rs, clipped to 1.
    
            | q = max(d_s)/d_rs, if q < 1
        p = |
            | 1, otherwise
        """

        ds = self.get_loudspeaker_distance(source=ref, r=r)
        max_ds = ds.max()

        drs = self.calc_distance(pos1=ref, pos2=source, r=r)

        q = max_ds/drs
        value = q if q < 1 else 1

        return value
    
    def get_k(self, d: float, p: float, b: float) -> float:

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
    
    def get_spatial_blur(self, ref: list) -> float:

        """
        calculate spatial blur

        r = (sum(d_ic)/N) + r_scalar 
        with 0.2 <= r_scalar <= 0.2
        """

        dist_cent = self.get_loudspeaker_distance(source=ref)
        rs = (np.sum(dist_cent)/len(dist_cent)) + 0.2
        return rs
    
    def calculate_gains(self, source: list, ref: list|None = None, spatial_blur: float|None = None) -> list[float]:

        """
        get loudloudspeaker gain factors
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

        eta = spat_blur/self.loudspeaker_pos.shape[1]

        d = self.get_loudspeaker_distance(source=source, r=spat_blur)
        p = self.get_p(source=source, ref=ref, r=spat_blur)
        b = self.get_b(ls_distance=d, p=p, eta=eta)
        k = self.get_k(d=d, p=p, b=b)
        w = self.w
        a = self.a

        v = (k * w * b)/d**a
        return v
    
    def display_panning(self, source: list|tuple):
        g = self.calculate_gains(source=source)
        self.plot_panning(source=source, g=g, base=None, mode="dbap")


    
        
