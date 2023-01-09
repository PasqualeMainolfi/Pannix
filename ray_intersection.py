"""
VBAP using line-line intersection

ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
"""

import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y
    
    def normalize(self):
        p = np.array([self.x, self.y], dtype=float)
        m = np.sqrt(p.dot(p))
        mag = m if m != 0 else 1
        p /= mag
        self.x = p[0]
        self.y = p[1]
    
    def __repr__(self) -> str:
        return f"\nclass Point: ({self.x}, {self.y})\n"

class Ray:
    def __init__(self, origin: list = [0, 0]):
        
        """
        origin: list, set origin (cartesian coordinate)
        """

        self.origin = Point(origin[0], origin[1])
        self.position = Point()

    def set_position(self, pos: list):

        """
        set source position

        pos: list, [x, y] (cartesian coordinate)
        """
        
        p = Point(pos[0], pos[1])
        self.position.x = p.x - self.origin.x
        self.position.y = p.y - self.origin.y
        self.position.normalize()
    
    def get_intersection(self, segment: list[list]):

        """
        get intersection point

        segment: list[list], [[p1x, p2x], [p1y, p2y]] pair of loudspeaker position in cartesian coordinate
        """

        source = np.array(segment, dtype=float)

        x1, y1 = source[0, 0], source[1, 0] 
        x2, y2 = source[0, 1], source[1, 1] 
        x3, y3 = self.origin.x, self.origin.y
        x4, y4 = self.origin.x + self.position.x, self.origin.y + self.position.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if den == 0:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))/den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2))/den

        if (0 <= t <= 1) and (0 <= u <= 1):
            xi = x1 + t * (x2 - x1)
            yi = y1 + t * (y2 - y1)
            intersect_point = Point(xi, yi)
            return intersect_point
        
        return None
    
    def show_intersection(self, source: list, segment: list[list]):

        p = self.get_intersection(segment=segment)
        
        plt.plot(segment[0, :], segment[1, :], "-o")

        if p:
            plt.scatter(p.x, p.y)

        xpos = [self.origin.x, self.position.x]
        ypos = [self.origin.y, self.position.y]
        plt.plot(xpos, ypos)
        plt.scatter(self.position.x, self.position.y, c="r")
        plt.show()

    def __repr__(self) -> str:
        s = f"\nclass Ray: pos = {self.position.x, self.position.y}\ndirection = {self.direction.x, self.direction.y}\n"
        return s