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
    
    def __repr__(self) -> str:
        return f"\nclass Point: ({self.x}, {self.y})\n"

class Ray:
    def __init__(self, start_point: list = [0, 0]):
        
        """
        start_point: list, set start_point origin (cartesian coordinate)
        """

        self.start_point = Point(start_point[0], start_point[1])
        self.end_point = Point()

    def set_source_position(self, pos: list):

        """
        set source position

        pos: list, [x, y] (cartesian coordinate)
        """
        
        self.end_point.x = pos[0]
        self.end_point.y = pos[1]
    
    def get_intersection(self, segment: list[list]):

        """
        get intersection point

        segment: list[list], [[p1x, p2x], [p1y, p2y]] pair of loudspeaker end_point in cartesian coordinate
        """

        source = np.array(segment, dtype=float)

        x1, y1 = source[0, 0], source[1, 0] 
        x2, y2 = source[0, 1], source[1, 1] 
        x3, y3 = self.start_point.x, self.start_point.y
        x4, y4 = self.end_point.x, self.end_point.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if den == 0:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))/den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2))/den

        if (0 <= t <= 1) and (u > 0):
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

        xpos = [self.start_point.x, self.end_point.x]
        ypos = [self.start_point.y, self.end_point.y]
        plt.plot(xpos, ypos)
        plt.scatter(self.end_point.x, self.end_point.y, c="r")
        plt.show()

    def __repr__(self) -> str:
        s = f"\nclass Ray: pos = {self.end_point.x, self.end_point.y}\ndirection = {self.direction.x, self.direction.y}\n"
        return s