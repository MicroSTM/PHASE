from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely import affinity
import math


def create_agent_shape(pos, angle, size):
    """create agent shape"""
    c_x, c_y = pos
    R = size
    angle = math.degrees(angle)
    polygon = Polygon([(c_x-(1.5*R), c_y-R), (c_x+(1.5*R), c_y-R), (c_x+(0.5*R), c_y+R), (c_x-(0.5*R), c_y+R),  \
                       (c_x-(1.5*R), c_y-R)])
    polygon = affinity.rotate(polygon, angle)
    return polygon


def create_item_shape(pos, size):
    """create object shape"""
    c_x, c_y = pos
    r = size
    circle = Point(c_x, c_y).buffer(r)
    return circle