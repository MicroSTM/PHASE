from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import pickle
import functools

os.environ["SDL_VIDEODRIVER"] = "dummy"  # https://www.pygame.org/wiki/DummyVideoDriver

import numpy as np
import random
import pygame
import Box2D
from Box2D.b2 import world, circleShape, edgeShape, polygonShape, dynamicBody
from Box2D import (
    b2DistanceJointDef,
    b2WeldJointDef,
    b2FrictionJointDef,
    b2FixtureDef,
    b2PolygonShape,
    b2ContactListener,
    b2Fixture,
    b2Vec2,
    b2RayCastOutput,
    b2RayCastInput,
    b2RayCastCallback,
)
from scipy.misc import imresize
from PIL import Image
import cv2
from datetime import datetime
import math
from utils import bfs
import pylab as plt
from random import randrange
import types
import copy
import scipy.ndimage as ndimage
import itertools
import pdb
import time
import matplotlib.pyplot as plt

from utils import *


LM_COLOR_LIST = [
    [45, 45, 180, 50],  # blue
    [50, 150, 50, 50],  # green
    [180, 45, 45, 50],  # red
    [180, 180, 45, 50],  # yellow
    [0.7 * 255, 0.2 * 255, 0.7 * 255, 255],  # purple
    [0.2 * 255, 0.2 * 255, 0.2 * 255, 255],  # black
]

ENT_COLOR_LIST = [
    [255, 0, 0, 255],  # red
    [0, 255, 0, 255],  # green
    [117, 216, 230, 255],  # blue
    [255, 153, 153, 255],  # pink
]

SIZE = [1 * 0.8, 1.5 * 0.8, 2 * 0.8, 1.01]

DENSITY = [1, 2]

STRENGTH = [150 / 5, 300 / 5, 450 / 5, 600 / 5]

POS = [
    (16 - 8, 12 + 8),  # 0
    (16 - 8, 12 + 2),  # 1
    (16 - 8, 12 - 2),  # 2
    (16 - 8, 12 - 8),  # 3
    (16 + 8, 12 + 8),  # 4
    (16 + 8, 12 + 2),  # 5
    (16 + 8, 12 - 2),  # 6
    (16 + 8, 12 - 8),  # 7
    (16, 12 + 4),  # 8
    (16, 12 - 4),  # 9
    (16 - 2, 12 + 4),  # 10
    (16 - 2, 12 - 4),  # 11
    (16 + 2, 12 + 4),  # 12
    (16 + 2, 12 - 4),  # 13
    (16 - 2, 12 + 8),  # 14
    (16 - 2, 12 - 8),  # 15
    (16 + 2, 12 + 8),  # 16
    (16 + 2, 12 - 8),  # 17
    (16, 12),  # 18
    (16 + 2, 12 + 2),  # 19
]

DECAY = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

COST = [0.0, 0.1, 0.2]

WALL_SEGS = {
    "..........": None,
    ".....=====": [5, 10],
    "=====.....": [0, 5],
    "...=======": [2.5, 10],
    "=======...": [0, 7.5],
    "==========": [0, 10],
}

FRICTION = 0
LM_SIZE = 2.5

RELATION = [0, +1, -1, +2, -2]


def _my_draw_edge(edge, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    vertices = [(body.transform * v) * PPM for v in edge.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.line(screen, color, vertices[0], vertices[1], 5)


edgeShape.draw = _my_draw_edge


def _my_draw_circle(circle, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(
        screen, color, [int(x) for x in position], int(circle.radius * PPM)
    )


circleShape.draw = _my_draw_circle


def my_draw_polygon(polygon, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    # draw body (polygon)
    Vs = polygon.vertices
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)
    # draw eyes (circles)
    R = _get_dist(Vs[0], Vs[1]) * 0.15
    eye_pos = [
        (
            (Vs[0][0] - Vs[1][0]) * 0.0 + Vs[1][0],
            (Vs[0][1] - Vs[1][1]) * 0.25 + Vs[1][1],
        ),
        (
            (Vs[3][0] - Vs[2][0]) * 0.0 + Vs[2][0],
            (Vs[3][1] - Vs[2][1]) * 0.25 + Vs[2][1],
        ),
    ]
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(
            screen, [0, 0, 0, 255], [int(x) for x in position], int(R * PPM)
        )

    R = _get_dist(Vs[0], Vs[1]) * 0.13
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(
            screen, [255, 255, 255, 255], [int(x) for x in position], int(R * PPM)
        )

    R = _get_dist(Vs[0], Vs[1]) * 0.1
    eye_pos = [
        (
            (Vs[0][0] - Vs[1][0]) * 0.0 + Vs[1][0],
            (Vs[0][1] - Vs[1][1]) * 0.15 + Vs[1][1],
        ),
        (
            (Vs[3][0] - Vs[2][0]) * 0.0 + Vs[2][0],
            (Vs[3][1] - Vs[2][1]) * 0.15 + Vs[2][1],
        ),
    ]
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(
            screen, [0, 0, 0, 255], [int(x) for x in position], int(R * PPM)
        )


polygonShape.draw = my_draw_polygon


def _my_draw_patch(pos, screen, color, PPM, SCREEN_WIDTH, SCREEN_HEIGHT):
    position = [(pos[0] - 2.5) * PPM, (pos[1] + 2.5) * PPM]
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.rect(
        screen, color, [int(x) for x in position] + [int(5 * PPM), int(5 * PPM)]
    )


def _get_world_pos(body):
    """get the position"""
    if isinstance(body.fixtures[0].shape, b2PolygonShape):
        center = (
            np.mean([v[0] for v in body.fixtures[0].shape.vertices]),
            np.mean([v[1] for v in body.fixtures[0].shape.vertices]),
        )
    else:
        center = body.fixtures[0].shape.pos
    position = body.transform * center
    # return position
    return (position[0], position[1])


def _get_world_vel(body):
    """get the velocity"""
    # vel = body.transform * body.linearVelocity
    # print(body.linearVelocity, vel)
    vel = body.linearVelocity
    return (vel[0], vel[1])


def _get_state_from_body(body):
    state = {
        "pos": _get_world_pos(body),
        "angle": body.angle,
        "vel": (body.linearVelocity.x, body.linearVelocity.y),
        "angleVel": body.angularVelocity,
        "attached": body.userData[-1],
    }
    return state


def _get_pos(body):
    """get the position"""
    position = body.transform * body.fixtures[0].shape.pos
    return position


def _get_body_bound(body):
    """get the boundary of a cicle"""
    position = body.transform * body.fixtures[0].shape.pos
    radius = body.fixtures[0].shape.radius
    return (
        position[0] - radius,
        position[1] - radius,
        position[0] + radius,
        position[1] + radius,
    )


def _get_door(body):
    vertices1 = [(body.transform * v) for v in body.fixtures[0].shape.vertices]
    vertices2 = [(body.transform * v) for v in body.fixtures[-1].shape.vertices]

    return [vertices1[0], vertices2[-1]]


"""TODO: currently only consider a rectangle w/o rotation"""


@functools.lru_cache(maxsize=64)
def _get_room_bound(body):
    """get the boundary of a room (upper-left corner + bottom-right corner)"""
    x_list, y_list = [], []
    for fixture in body.fixtures:
        vertices = [(body.transform * v) for v in fixture.shape.vertices]
        x_list += [v[0] for v in vertices]
        y_list += [v[1] for v in vertices]
    min_x, min_y = min(x_list), min(y_list)
    max_x, max_y = max(x_list), max(y_list)
    return (min_x, min_y, max_x, max_y)


def _in_room(body, room):
    body_bound = _get_body_bound(body)
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return (
        body_bound[2] >= min_x
        and body_bound[3] >= min_y
        and body_bound[0] <= max_x
        and body_bound[1] <= max_y
    )


@functools.lru_cache(maxsize=64)
def _point_in_room(point, room):
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return (
        point[0] >= min_x
        and point[1] >= min_y
        and point[0] <= max_x
        and point[1] <= max_y
    )


def _get_obs(screen, SCREEN_WIDTH, SCREEN_HEIGHT):
    string_image = pygame.image.tostring(screen, "RGB")
    temp_surf = pygame.image.fromstring(
        string_image, (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB"
    )
    return pygame.surfarray.array3d(temp_surf)


# @functools.lru_cache(maxsize=64)
def _get_dist(pos1, pos2):
    """get the distance between two 2D positions"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def _get_dist_lm(pos, lm):
    if pos[0] > lm[0] - 1.5 and pos[0] < lm[0] + 1.5:
        if pos[1] > lm[1] - 1.5 and pos[1] < lm[1] + 1.5:
            return 0
        else:
            return min(abs(pos[1] - (lm[1] - 2.5)), abs(pos[1] - (lm[1] + 2.5)))
    elif pos[1] > lm[1] - 1.5 and pos[1] < lm[1] + 1.5:
        return min(abs(pos[0] - (lm[0] - 2.5)), abs(pos[0] - (lm[0] + 2.5)))
    else:
        return _get_dist(pos, (lm[0] - 2.5, lm[1] - 2.5))


def _norm(vec):
    """L2-norm of a vector"""
    return (vec[0] ** 2 + vec[1] ** 2) ** 0.5


def _get_angle(vec1, vec2):
    """angle between 2 vectors"""
    return np.arccos(
        ((vec1[0] * vec2[0]) + (vec1[1] * vec2[1])) / (_norm(vec1) * _norm(vec2))
    )


def _get_segment_intersection(pt1, pt2, ptA, ptB):
    # https://www.cs.hmc.edu/ACM/lectures/intersections.html
    """this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
        valid == 0 if there are 0 or inf. intersections (invalid)
        valid == 1 if it has a unique intersection ON the segment"""

    DET_TOLERANCE = 0.00000001
    # the first line is pt1 + r*(pt2-pt1)
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1
    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y
    DET = -dx1 * dy + dy1 * dx
    if math.fabs(DET) < DET_TOLERANCE:
        return (0, 0, 0, 0, 0)
    # now, the determinant should be OK
    DETinv = 1.0 / DET
    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))
    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))
    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2 * dy - dx
    y = 0
    for x in range(dx + 1):
        # for x in range(int(np.floor(dx + 1))):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


# @functools.lru_cache(maxsize=64)
def _get_point_dist_from_seg(p1, p2, pnt):
    p1, p2, pnt = np.array(p1), np.array(p2), np.array(pnt)
    return np.abs(np.cross(p2 - p1, p1 - pnt) / _norm(p2 - p1))


def _not_door(x, y, doors_pos):
    if x == 16:
        if y > doors_pos[0][1] or y < doors_pos[2][1]:
            return False
    if y == 12:
        if x < doors_pos[1][0] or x > doors_pos[3][0]:
            return False
    return True


def _on_room_boundary(grid_cell, room):
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return (
        grid_cell[0] == 0
        or grid_cell[1] == 0
        or grid_cell[1] == max_x - min_x - 1
        or grid_cell[1] == max_y - min_y - 1
    )


def flip_pos(x, y, flip, center=(16, 12)):
    c1, c2 = center
    if flip == 0:
        y = (c2 - y) + c2
    elif flip == 1:
        x = (c1 - x) + c1
    elif flip == 2:
        x, y = (y - c2) + c1, (x - c1) + c2
    elif flip == 3:
        x, y = -(y - c2) + c1, -(x - c1) + c2
    return x, y


def flip_positions(positions, flip):
    flipped_positions = []
    for pos in positions:
        flipped_pos = flip_pos(pos[0], pos[1], flip)
        flipped_positions.append(flipped_pos)
    return flipped_positions


def flip_angles(angles, flip):
    if flip == 0:
        flipped_angles = [np.pi - a for a in angles]
    elif flip == 1:
        flipped_angles = [-a for a in angles]
    # convert radian to unit vector, flip like pos, convert back to radian
    elif flip == 2 or flip == 3:
        flipped_angles = []
        for a in angles:
            x, y = np.cos(a), np.sin(a)
            x, y = flip_pos(x, y, flip, center=(0, 0))
            flipped_a = math.atan2(y, x)
            flipped_angles.append(flipped_a - np.pi)  # TODO pi?
    return flipped_angles


def flip_goal(goal, flip):
    lm_order = list(range(4))
    if flip == 0:
        lm_order = lm_order[::-1]
    elif flip == 1:
        lm_order = lm_order[:2][::-1] + lm_order[2:][::-1]
    elif flip == 2:
        lm_order = [lm_order[2], lm_order[1], lm_order[0], lm_order[3]]
    elif flip == 3:
        lm_order = [lm_order[0], lm_order[3], lm_order[2], lm_order[1]]
    flipped_goal = copy.deepcopy(goal)
    if "LMO" in goal or "LMA" in goal:
        flipped_goal[2] = lm_order[goal[2]]
    return flipped_goal


def flip_maze(maze_def, flip):
    if flip == 0:
        maze_def = [maze_def[2], maze_def[1], maze_def[0], maze_def[3]]
    elif flip == 1:
        maze_def = [maze_def[0], maze_def[3], maze_def[2], maze_def[1]]
    elif flip == 2:
        maze_def = [maze_def[3], maze_def[2], maze_def[1], maze_def[0]]
    elif flip == 3:
        maze_def = [maze_def[1], maze_def[0], maze_def[3], maze_def[2]]
    return maze_def


# Belief can represent an agent's belief over:
# the env state (partially observable) = pos, shape, vel (lin & ang), angle for any entity
# other agent observations = pos, shape, vel, angle for each entity
# goals of others = probability of each goal in GOAL
class Belief:
    def __init__(
        self, room_dim, world_center, agent_id, self_state, num_entities, n_particles
    ):
        """sample init pos,shape from uniform distribution, assume 0 initial vel,ang
        each particle has the state of the world"""
        self.n_particles = n_particles
        self.room_dim, self.world_center = room_dim, world_center
        self.particles = [
            [None for _ in range(num_entities)] for _ in range(n_particles)
        ]
        x, y = np.meshgrid(
            list(
                range(
                    int(world_center[0] - (room_dim[0]) // 2 + 2),
                    int(world_center[0] + (room_dim[0]) // 2 - 1),
                )
            ),
            list(
                range(
                    int(world_center[1] - (room_dim[1]) // 2 + 2),
                    int(world_center[1] + (room_dim[1]) // 2 - 1),
                )
            ),
        )
        self.init_pos = list(zip(*(x.flatten(), y.flatten())))
        for world_state in self.particles:
            for entity_idx in range(num_entities):
                if entity_idx == agent_id:
                    p = copy.deepcopy(self_state)
                else:
                    p = {}
                    pos = random.choice(self.init_pos)
                    p["pos"] = (int(pos[0]), int(pos[1]))
                    p["angle"] = 0.0  # math.radians(np.random.uniform(-360,360))
                    p["vel"] = [0.0, 0.0]  # lin.x, lin.y
                    p["angleVel"] = 0.0
                    p["attached"] = None
                world_state[entity_idx] = p
        self.curr_belief = None  # estimated state

    # different update function for env and goals? - for now assume goals are known.
    # TODO use map of what I can see, estimate other's gaze.
    def update(self, next_particles):
        self.particles = next_particles

    def sample_random_state(self):
        """return random particle"""
        self.curr_belief = random.choice(self.particles)
        return copy.deepcopy(self.curr_belief)

    def sample_dist_state(self):
        """return particle sampled from joint pos distribution"""
        all_pos = [
            np.array(
                [[np.floor(p["pos"][0]), np.floor(p["pos"][1])] for p in world_state]
            ).flatten()
            for world_state in self.particles
        ]
        unique_pos, cnt = np.unique(all_pos, return_counts=True, axis=0)
        self.curr_belief = random.choices(unique_pos, weights=cnt / np.sum(cnt), k=1)
        return copy.deepcopy(self.curr_belief)

    def sample_freq_state(self):
        """return (one of the) most common particle for joint pos distribution"""
        all_pos = [
            np.array(
                [[np.floor(p["pos"][0]), np.floor(p["pos"][1])] for p in world_state]
            ).flatten()
            for world_state in self.particles
        ]
        unique_pos, cnt = np.unique(all_pos, return_counts=True, axis=0)
        most_common = cnt == np.max(cnt)
        self.curr_belief = random.choice(unique_pos[most_common])
        return copy.deepcopy(self.curr_belief)

    def resample_one_entity(self, entity_id):
        """randomly resample a specified entity for all particles"""
        for state in self.particles:
            p = {}
            pos = random.choice(self.init_pos)
            p["pos"] = (int(pos[0]), int(pos[1]))
            p["angle"] = 0.0  # math.radians(np.random.uniform(-360,360))
            p["vel"] = [0.0, 0.0]  # lin.x, lin.y
            p["angleVel"] = 0.0
            p["attached"] = None
            state[entity_id] = p


class RayCastAnyCallback(b2RayCastCallback):
    """This callback finds any hit"""

    def __repr__(self):
        return "Any hit"

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False
        self.point = None
        self.normal = None

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        return 0.0


class MyContactListener(b2ContactListener):
    def agent_item_update(self, agent, item, manifold):
        agent_id = agent[0][1]
        if item[1] == agent_id:  # item attached to agent
            agent[1][agent_id][item[0]] = [
                manifold,
                True,
                False,
            ]  # [contact, is item attached to agent, does contact involve attached item]
        elif item[1] is None:  # item not attached
            agent[1][agent_id][item[0]] = [manifold, False, False]
        else:  # attached to other agent
            other_agent_id = 1 - agent_id  # TODO: remove this for a more general setup
            agent[1][agent_id][item[0]] = [
                manifold,
                False,
                True,
            ]  # is touching item that's attached to other agent
            agent[1][other_agent_id][agent[0]] = [
                manifold,
                False,
                True,
            ]  # is agent touching through attached item

    def agent_item_remove(self, agent, item):
        agent_id = agent[0][1]
        if item[1] == agent_id:  # item attached to agent
            return
        elif item[1] is None:  # item not attached
            agent[1][agent_id].pop(item[0])
        else:  # attached to other agent
            other_agent_id = 1 - agent_id  # TODO: remove this for a more general setup
            agent[1][agent_id].pop(item[0])
            agent[1][other_agent_id].pop(agent[0])

    def agent_agent_update(self, agent1, agent2, manifold):
        # update both agents
        agent1[1][agent1[0][1]][agent2[0]] = [manifold, False, False]
        agent2[1][agent2[0][1]][agent1[0]] = [manifold, False, False]

    def agent_agent_remove(self, agent1, agent2):
        # update both agents
        if (
            agent2[0] not in agent1[1][agent1[0][1]]
            or agent1[0] not in agent2[1][agent2[0][1]]
        ):
            print("contact sensor")  # TODO
            return
        agent1[1][agent1[0][1]].pop(agent2[0])
        agent2[1][agent2[0][1]].pop(agent1[0])

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData
        if (
            bodyA is not None and bodyB is not None
        ):  # both not None means collision between agents and/or items
            if bodyA[0][0] == "agent" and bodyB[0][0] == "item":
                self.agent_item_update(bodyA, bodyB, contact.manifold)
            elif bodyA[0][0] == "item" and bodyB[0][0] == "agent":
                self.agent_item_update(bodyB, bodyA, contact.manifold)
            elif bodyA[0][0] == "agent" and bodyB[0][0] == "agent":
                self.agent_agent_update(bodyA, bodyB, contact.manifold)

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData
        if bodyA is not None and bodyB is not None:
            if bodyA[0][0] == "agent" and bodyB[0][0] == "item":
                self.agent_item_remove(bodyA, bodyB)
            elif bodyA[0][0] == "item" and bodyB[0][0] == "agent":
                self.agent_item_remove(bodyB, bodyA)
            elif bodyA[0][0] == "agent" and bodyB[0][0] == "agent":
                self.agent_agent_remove(bodyA, bodyB)


class Maze_v1:
    """two agents move one item to certain position"""

    def __init__(
        self,
        action_type,
        maze_sampler,
        goals,
        strengths,
        sizes,
        densities,
        init_positions,
        action_space_types,
        costs,
        temporal_decay,
        visibility,
        num_agents=2,
        num_items=2,
        all_directions=False,
        PPM=20.0,
        TARGET_FPS=60,
        SCREEN_WIDTH=640,
        SCREEN_HEIGHT=480,
        TIME_STEP=5,
        enable_renderer=True,
        render_separate_particles=False,
        random_colors=False,
        random_colors_agents=False,
        full_obs=[0, 0],
        n_particles=50,
        belief_prior=0.5,
        init_agent_angles=[0.0, 0.0],
        use_all_init_grid_pos=False,
        human_simulation=False,
    ):
        self.action_type = action_type
        self.maze_sampler = maze_sampler
        self.goals = list(goals)
        self.strengths = list(strengths)
        self.strengths_value = [STRENGTH[s] for s in strengths]
        self.sizes = list(sizes)
        self.sizes_value = [SIZE[s] for s in sizes]
        self.densities = list(densities)
        self.init_positions = list(init_positions)
        self.action_space_types = action_space_types
        self.costs = list(costs)
        self.visibility = list(visibility)
        self.temporal_decay = [DECAY[td] for td in temporal_decay]
        self.PPM = PPM
        self.TARGET_FPS = TARGET_FPS
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.TIME_STEP = 1.0 / TARGET_FPS  # * TIME_STEP
        self.NUM_STEPS_PER_TICK = TIME_STEP
        self.enable_renderer = enable_renderer
        self.render_separate_particles = render_separate_particles
        self.obs_dim = (3, 86, 86)
        self.state_dim = 28
        self.num_agents = num_agents
        self.num_items = num_items
        self.num_entities = num_agents + num_items
        self.random_colors = random_colors
        self.random_colors_agents = random_colors_agents
        self.full_obs = full_obs
        self.n_particles = n_particles
        self.belief_prior = belief_prior
        self.init_agent_angles = init_agent_angles
        self.use_all_init_grid_pos = use_all_init_grid_pos
        self.human_simulation = human_simulation

        if all_directions:
            self.action_space = [
                "turnleft",
                "turnright",
                "up",
                "down",
                "left",
                "right",
                "upleft",
                "upright",
                "downleft",
                "downright",
                "stop",
                "noforce",
                "attach",
                "detach",
            ]
        else:
            self.action_space = [
                "turnleft",
                "turnright",
                "up",
                "stop",
                "noforce",
                "attach",
                "detach",
            ]
            # self.action_space = ['turnleft', 'turnright', 'up', 'down', 'left', 'right', 'stop', 'noforce', 'attach', 'detach']
        self.action_size = len(self.action_space)
        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption("Maze_v1")
            self.clock = pygame.time.Clock()
        self.room_dim = (20, 20)
        self._max_dist = _norm((20, 20))
        self.door_length = 0
        self.clip_interval = None
        self.touch_sensor = [{} for _ in range(num_agents)]

        self.viz_map = None

        random.seed(1)

    def seed(self):
        random.seed(datetime.now())

    def set_clip_interval(self, clip_interval):
        self.clip_interval = list(clip_interval)

    def _get_room_id(self, pos):
        if pos[0] < 16:
            if pos[1] > 12:
                return 0
            else:
                return 3
        else:
            if pos[1] > 12:
                return 1
            else:
                return 2

    def get_action_id(self, action):
        for action_id, a in enumerate(self.action_space):
            if action == a:
                return action_id
        return 0

    def setup(
        self,
        env_id,
        agent_id,
        max_episode_length,
        record_path=None,
        contact_listen=False,
        agent_order=None,
        item_order=None,
        landmark_color_order=None,
        flip_env=None,
        rotate_env=False,
    ):
        """setup a new espisode"""
        self.env_id = env_id
        self.planning_agent = agent_id
        self.max_episode_length = max_episode_length
        self.flip_env = flip_env
        self.rotate_env = rotate_env
        if contact_listen:
            self.world = world(
                contactListener=MyContactListener(), gravity=(0, 0), doSleep=True
            )
        else:
            self.world = world(gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
                (-self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, self.room_dim[1] / 2 - self.door_length),
            ]
        )
        self.connected = [[None] * 4 for _ in range(4)]
        self.connected[0][1] = self.connected[1][0] = 0
        self.connected[0][3] = self.connected[3][0] = 1
        self.connected[2][3] = self.connected[3][2] = 2
        self.connected[1][2] = self.connected[2][1] = 3

        self.doors_pos = [
            (16, self.room_dim[1] / 4 + 12),
            (-self.room_dim[0] / 4 + 16, 12),
            (16, -self.room_dim[1] / 4 + 12),
            (self.room_dim[0] / 4 + 16, 12),
        ]  # middle door pos
        self.doors_size = [10] * 4

        # build maze
        self.wall_segs = []
        env_def = self.maze_sampler.get_env_def(env_id)
        maze_def = env_def["maze_def"]
        for wall_id, wall in enumerate(maze_def):
            # print(wall_id, wall, seg, env_id, env_def['maze_def'])
            if WALL_SEGS[wall] is not None:
                seg = [WALL_SEGS[wall][0], WALL_SEGS[wall][1]]
                self.doors_size[wall_id] = 10 - abs(seg[0] - seg[1])
                if wall_id == 0:
                    # if env_id == 16:
                    #     seg[1] -= 3
                    self.room.CreateEdgeChain([(0, seg[0]), (0, seg[1])])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[0][1] = None
                        self.connected[1][0] = None
                    else:
                        self.doors_pos[0] = (
                            16,
                            12 + (self.room_dim[1] / 2 + seg[1]) / 2,
                        )
                    self.wall_segs.append([(16, seg[0] + 12), (16, seg[1] + 12)])
                elif wall_id == 1:
                    self.room.CreateEdgeChain([(-seg[0], 0), (-seg[1], 0)])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[0][3] = None
                        self.connected[3][0] = None
                    else:
                        self.doors_pos[1] = (
                            16 + (-self.room_dim[0] / 2 - seg[1]) / 2,
                            12,
                        )
                    self.wall_segs.append([(16 - seg[0], 12), (16 - seg[1], 12)])
                elif wall_id == 2:
                    self.room.CreateEdgeChain([(0, -seg[0]), (0, -seg[1])])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[2][3] = None
                        self.connected[3][2] = None
                    else:
                        self.doors_pos[2] = (
                            16,
                            12 + (-self.room_dim[1] / 2 - seg[1]) / 2,
                        )
                    self.wall_segs.append([(16, -seg[0] + 12), (16, -seg[1] + 12)])
                elif wall_id == 3:
                    self.room.CreateEdgeChain([(seg[0], 0), (seg[1], 0)])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[1][2] = None
                        self.connected[2][1] = None
                    else:
                        self.doors_pos[3] = (
                            16 + (self.room_dim[0] / 2 + seg[1]) / 2,
                            12,
                        )
                    self.wall_segs.append([(16 + seg[0], 12), (16 + seg[1], 12)])

        self.doors_pos_start = [
            (16, (self.room_dim[1] // 2) - self.doors_size[0] + 12),
            (-(self.room_dim[0] // 2) + self.doors_size[1] + 16, 12),
            (16, -(self.room_dim[1] // 2) + self.doors_size[2] + 12),
            ((self.room_dim[0] // 2) - self.doors_size[3] + 16, 12),
        ]

        self.room_bound = _get_room_bound(self.room)

        self.agents, self.items = [], []
        self.init_agents_pos = [None] * self.num_agents
        self.init_items_pos = [None] * self.num_items
        self.trajectories = [None] * (self.num_agents + self.num_items + 1)
        self.tmp_trajectories = [
            [] for _ in range(self.num_agents + self.num_items + 1)
        ]

        room_dim, world_center = self.room_dim, self.room.position
        x, y = np.meshgrid(
            list(
                range(
                    int(world_center[0] - (room_dim[0]) // 2 + 2),
                    int(world_center[0] + (room_dim[0]) // 2 - 2),
                )
            ),
            list(
                range(
                    int(world_center[1] + (room_dim[1]) // 2 - 2),
                    int(world_center[1] - (room_dim[1]) // 2 + 2),
                    -1,
                )
            ),
        )
        self.all_init_pos = list(zip(*(x.flatten(), y.flatten())))
        # add agents
        for agent_id in range(self.num_agents):
            if self.use_all_init_grid_pos:
                gx, gy = (
                    self.all_init_pos[self.init_positions[agent_id]][0],
                    self.all_init_pos[self.init_positions[agent_id]][1],
                )
                x = random.uniform(gx, gx + 1)
                y = random.uniform(gy - 1, gy)
            else:
                x, y = (
                    POS[self.init_positions[agent_id]][0],
                    POS[self.init_positions[agent_id]][1],
                )
            self.init_agents_pos[agent_id] = (x, y)
            R = SIZE[self.sizes[agent_id]]
            body = self.world.CreateDynamicBody(
                position=(x, y),
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(
                        vertices=[
                            (-1.5 * R, -R),
                            (-0.5 * R, R),
                            (0.5 * R, R),
                            (1.5 * R, -R),
                        ]
                    ),
                    density=DENSITY[self.densities[agent_id]],
                    friction=FRICTION,
                ),
                angle=self.init_agent_angles[agent_id],
            )
            body.userData = [
                ("agent", agent_id),
                self.touch_sensor,
                None,
            ]  # 3rd entry for attached item_id (None if not attached)
            body.field_of_view = np.zeros(self.room_dim)
            body.certainty = np.zeros(self.room_dim)
            body.last_observations = {}
            random.seed(datetime.now())
            body.beliefs = Belief(
                room_dim=self.room_dim,
                world_center=self.room.position,
                agent_id=agent_id,
                self_state=_get_state_from_body(body),
                num_entities=self.num_agents + self.num_items,
                n_particles=self.n_particles,
            )
            self.agents.append(body)
            self.trajectories[agent_id] = [
                (x, y, 0, 0, self.init_agent_angles[agent_id])
            ]

        # add items
        ITEM_BASE = self.num_agents
        for item_id in range(self.num_items):
            index = ITEM_BASE + item_id
            # print(self.init_positions, index)
            if self.use_all_init_grid_pos:
                gx, gy = (
                    self.all_init_pos[self.init_positions[index]][0],
                    self.all_init_pos[self.init_positions[index]][1],
                )
                x = random.uniform(gx, gx + 1)
                y = random.uniform(gy - 1, gy)
            else:
                x, y = (
                    POS[self.init_positions[index]][0],
                    POS[self.init_positions[index]][1],
                )
            self.init_items_pos = [(x, y)]
            body = self.world.CreateDynamicBody(position=(x, y))
            body.CreateCircleFixture(
                radius=SIZE[self.sizes[index]],
                density=DENSITY[self.densities[index]],
                friction=FRICTION,
                restitution=0,
            )
            body.userData = [
                ("item", item_id),
                None,
            ]  # 2nd entry for attached agent_id (None if not attached)
            self.items.append(body)
            self.trajectories[index] = [(x, y, 0, 0, 0)]

        # friction
        self.groundBody = self.world.CreateStaticBody(
            position=(16, 12),
            shapes=polygonShape(box=(self.room_dim[0], self.room_dim[1])),
        )
        for body in self.agents + self.items:
            dfn = b2FrictionJointDef(
                bodyA=body,
                bodyB=self.groundBody,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                maxForce=100,
                maxTorque=0,
            )
            self.world.CreateJoint(dfn)

        self.steps = 0
        self.running = False
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
            )
        self.attached = [None] * self.num_agents
        self.once_attached = [False] * self.num_agents

        self.landmark_centers = [
            (16 - self.room_dim[0] / 2 + 2.5, 12 + self.room_dim[1] / 2 - 2.5),
            (16 + self.room_dim[0] / 2 - 2.5, 12 + self.room_dim[1] / 2 - 2.5),
            (16 + self.room_dim[0] / 2 - 2.5, 12 - self.room_dim[1] / 2 + 2.5),
            (16 - self.room_dim[0] / 2 + 2.5, 12 - self.room_dim[1] / 2 + 2.5),
        ]
        self.landmark_corners = [
            (16 - self.room_dim[0] / 2 + 1.25, 12 + self.room_dim[1] / 2 - 1.25),
            (16 + self.room_dim[0] / 2 - 1.25, 12 + self.room_dim[1] / 2 - 1.25),
            (16 + self.room_dim[0] / 2 - 1.25, 12 - self.room_dim[1] / 2 + 1.25),
            (16 - self.room_dim[0] / 2 + 1.25, 12 - self.room_dim[1] / 2 + 1.25),
        ]

        # if landmark_color_order is not None:
        #     self.landmark_color_code = {
        #         landmark_color_order.index(0): 0,
        #         landmark_color_order.index(1): 1,
        #         landmark_color_order.index(2): 2,
        #         landmark_color_order.index(3): 3
        #     }
        #     self.landmark_colors = [
        #         LM_COLOR_LIST[landmark_color_order.index(0)],
        #         LM_COLOR_LIST[landmark_color_order.index(1)],
        #         LM_COLOR_LIST[landmark_color_order.index(2)],
        #         LM_COLOR_LIST[landmark_color_order.index(3)]
        #     ]
        # else:
        #     self.landmark_colors = LM_COLOR_LIST[:4]

        # agent_order=None, item_order=None, landmark_color_order=None
        if agent_order is not None and item_order is not None:
            self.colors = {
                0: (0, 0, 0, 255),  # ground body
                agent_order[0]: ENT_COLOR_LIST[0],
                agent_order[1]: ENT_COLOR_LIST[1],
                item_order[0]: ENT_COLOR_LIST[2],
                item_order[1]: ENT_COLOR_LIST[3],
            }
            self.entity_color_code = {
                agent_order[0] - 1: 0,
                agent_order[1] - 1: 1,
                item_order[0]: 2,
                item_order[1]: 3,
            }
        else:
            if not self.random_colors:
                if not self.random_colors_agents:
                    self.colors = {
                        0: (0, 0, 0, 255),  # ground body
                        1: ENT_COLOR_LIST[0],  # agent 1
                        2: ENT_COLOR_LIST[1],  # agent 2
                        3: ENT_COLOR_LIST[2],  # item 1
                        4: ENT_COLOR_LIST[3],  # item 2
                    }
                    self.entity_color_code = {
                        0: 0,
                        1: 1,
                        2: 2,
                        3: 3,
                    }
                else:
                    order = [1, 2]
                    random.shuffle(order)
                    self.colors = {
                        0: (0, 0, 0, 255),  # ground body
                        order[0]: ENT_COLOR_LIST[0],  # agent 1
                        order[1]: ENT_COLOR_LIST[1],  # agent 2
                        3: ENT_COLOR_LIST[2],  # item 1
                        4: ENT_COLOR_LIST[3],  # item 2
                    }
                    self.entity_color_code = {
                        order[0] - 1: 0,
                        order[1] - 1: 1,
                        2: 2,
                        3: 3,
                    }
            else:
                order_1 = [1, 2]
                random.shuffle(order_1)
                order_2 = [3, 4]
                random.shuffle(order_2)
                self.colors = {
                    0: (0, 0, 0, 255),  # ground body
                    order_1[0]: ENT_COLOR_LIST[0],
                    order_1[1]: ENT_COLOR_LIST[1],
                    order_2[0]: ENT_COLOR_LIST[2],
                    order_2[1]: ENT_COLOR_LIST[3],
                }
                self.entity_color_code = {
                    order_1[0] - 1: 0,
                    order_1[1] - 1: 1,
                    order_2[0] - 1: 2,
                    order_2[1] - 1: 3,
                }

        self.path, self.doors = {}, {}
        for room_id1 in range(4):
            for room_id2 in range(4):
                if room_id1 != room_id2:
                    (
                        self.path[(room_id1, room_id2)],
                        self.doors[(room_id1, room_id2)],
                    ) = bfs(self.connected, room_id1, room_id2)
        # if self.viz_map is None:
        #     self._create_visibility_map()

    def start(self, replay=False, record_dir=None):
        """start the episode"""
        self.running = True
        self.repeat_actions = [None] * 2
        self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.actions = [None] * 2
        self.agent_states = [_get_state_from_body(body) for body in self.agents]
        self.item_states = [_get_state_from_body(item) for item in self.items]
        if not replay and not self.human_simulation:
            self.update_field_of_view()
            self.update_observations()
            self.belief_step([None, None])
        if self.human_simulation:
            self.update_observations_human_simulation()

        if self.use_all_init_grid_pos and record_dir is not None:
            # save can see
            visibility_X_door_pos = self.doors_pos_start
            visibility_X_agent_vs = [
                [
                    [
                        ag.GetWorldPoint(localPoint=v)[0],
                        ag.GetWorldPoint(localPoint=v)[1],
                    ]
                    for v in ag.fixtures[0].shape.vertices
                ]
                for ag in self.agents
            ]
            # print('visibility_X_agent_vs',visibility_X_agent_vs)
            visibility_X_item_pos_r = [
                [it["pos"][0], it["pos"][1], SIZE[s]]
                for it, s in zip(self.item_states, self.sizes[self.num_agents :])
            ]
            visibility_Y = [
                ("agent", 1) in self.agents[0].last_observations,
                ("agent", 0) in self.agents[1].last_observations,
            ]
            path = record_dir + "/visibility.pik"
            pickle.dump(
                [
                    visibility_X_door_pos,
                    visibility_X_agent_vs,
                    visibility_X_item_pos_r,
                    visibility_Y,
                ],
                open(path, "wb"),
            )

    def reset_history(self):
        self.repeat_actions = [None] * self.num_agents
        self.actions = [None] * self.num_agents
        self.trajectories = [
            [
                (
                    state["pos"][0],
                    state["pos"][1],
                    state["vel"][0],
                    state["vel"][1],
                    state["angle"],
                )
            ]
            for state in self.agent_states + self.item_states
        ]
        self.once_attached = [a for a in self.attached]

    def check_attachable(self, agent_id, item_id=None, threshold=0.5):
        """check if agent can grab an item"""
        """TODO: based on any given state instead of the ground-truth state"""
        if (
            self.action_space_types[agent_id] == 2
            or self.attached[agent_id] is not None
        ):
            return False
        head_mid_point = self.agents[agent_id].GetWorldPoint(
            localPoint=(0.0, SIZE[self.sizes[agent_id]])
        )
        tail_mid_point = self.agents[agent_id].GetWorldPoint(
            localPoint=(0.0, -SIZE[self.sizes[agent_id]])
        )
        right_mid_point = self.agents[agent_id].GetWorldPoint(
            localPoint=(SIZE[self.sizes[agent_id]], 0.0)
        )
        left_mid_point = self.agents[agent_id].GetWorldPoint(
            localPoint=(-SIZE[self.sizes[agent_id]], 0.0)
        )
        all_agent_pos = [
            head_mid_point,
            tail_mid_point,
            right_mid_point,
            left_mid_point,
        ]
        min_dist = 1e6
        selected_item_id = None
        selected_agent_anchor_idx = None
        if item_id is not None:
            all_dist = [
                _get_dist(pos, self.item_states[item_id]["pos"])
                for pos in all_agent_pos
            ]
            cur_dist = min(all_dist)
            agent_anchor_idx = all_agent_pos.index(
                all_agent_pos[all_dist.index(cur_dist)]
            )
            # print(cur_dist, SIZE[self.sizes[self.num_agents + item_id]] + 0.5, self.sizes, item_id, self.num_agents + item_id)
            if cur_dist < SIZE[self.sizes[self.num_agents + item_id]] + threshold:
                selected_item_id = item_id
                selected_agent_anchor_idx = agent_anchor_idx
        else:
            for item_id in range(self.num_items):
                all_dist = [
                    _get_dist(pos, self.item_states[item_id]["pos"])
                    for pos in all_agent_pos
                ]
                cur_dist = min(all_dist)
                agent_anchor_idx = all_agent_pos.index(
                    all_agent_pos[all_dist.index(cur_dist)]
                )
                # print(cur_dist, SIZE[self.sizes[self.num_agents + item_id]] + 0.2)
                if (
                    cur_dist < SIZE[self.sizes[self.num_agents + item_id]] + threshold
                    and cur_dist < min_dist
                ):
                    min_dist = cur_dist
                    selected_item_id = item_id
                    selected_agent_anchor_idx = agent_anchor_idx
        return selected_item_id is not None

    def _is_blocked(self, doors, size):
        for door in doors:
            if self.doors_size[door] < 2 * size:
                return True
        return False

    def get_action_space(self, agent_id):
        # print('get_action_space',self.goals[agent_id])
        num_actions = len(self.action_space) - self.action_space_types[agent_id]
        if not self.attached[agent_id]:
            num_actions = min(num_actions, len(self.action_space) - 1)
        # else:
        #     #don't restrict actions when attached
        #     return self.action_space[:num_actions]
        if self.goals[agent_id][0] in ["RO", "GE", "TE"]:
            if self.goals[agent_id][0] == "RO":
                item_room = self._get_room_id(self.item_states[0]["pos"])
                goal_room = self.goals[agent_id][2]
                if item_room != goal_room and self._is_blocked(
                    self.doors[(item_room, goal_room)], SIZE[self.sizes[2]]
                ):
                    num_actions = min(num_actions, len(self.action_space) - 1)
                if not self.attached[agent_id]:
                    num_actions = min(num_actions, len(self.action_space) - 1)
                else:
                    num_actions = min(num_actions, len(self.action_space) - 2)
            elif self.goals[agent_id][0] == "TE":
                return ["turnleft", "turnright", "up", "stop", "noforce"]
        elif self.goals[agent_id][0] == "LMO":
            num_actions = min(num_actions, len(self.action_space) - 1)
        elif self.goals[agent_id][0] == "RA" or (
            self.goals[agent_id][0] == "LMA" and self.goals[agent_id][1] == agent_id
        ):
            return ["turnleft", "turnright", "up", "stop", "noforce"]
        # elif self.attached[agent_id] is None and self.goals[agent_id][0]=='LMOF':
        #     return ['turnleft', 'turnright', 'up', 'down', 'stop', 'noforce']
        else:
            num_actions = min(num_actions, len(self.action_space) - 2)
        print("get_action_space", self.goals[agent_id], self.action_space[:num_actions])
        return self.action_space[:num_actions]

    def send_action(self, agent_id, action):
        """send action to an agent"""
        if action is None:
            return
        self.repeat_actions[agent_id] = action
        if action not in (
            self.action_space[: -self.action_space_types[agent_id]]
            if self.action_space_types[agent_id]
            else self.action_space
        ):
            return
        if action == "attach":
            head_mid_point = self.agents[agent_id].GetWorldPoint(
                localPoint=(0.0, SIZE[self.sizes[agent_id]])
            )
            tail_mid_point = self.agents[agent_id].GetWorldPoint(
                localPoint=(0.0, -SIZE[self.sizes[agent_id]])
            )
            right_mid_point = self.agents[agent_id].GetWorldPoint(
                localPoint=(SIZE[self.sizes[agent_id]], 0.0)
            )
            left_mid_point = self.agents[agent_id].GetWorldPoint(
                localPoint=(-SIZE[self.sizes[agent_id]], 0.0)
            )
            all_agent_pos = [
                head_mid_point,
                tail_mid_point,
                right_mid_point,
                left_mid_point,
            ]
            print(all_agent_pos)
            if self.attached[agent_id] is None:
                min_dist = 1e6
                selected_item_id = None
                selected_agent_anchor_idx = None
                for item_id in range(self.num_items):
                    all_dist = [
                        _get_dist(pos, self.item_states[item_id]["pos"])
                        for pos in all_agent_pos
                    ]
                    cur_dist = min(all_dist)
                    agent_anchor_idx = all_agent_pos.index(
                        all_agent_pos[all_dist.index(cur_dist)]
                    )
                    print(cur_dist, SIZE[self.sizes[self.num_agents + item_id]] + 0.2)
                    if (
                        cur_dist < SIZE[self.sizes[self.num_agents + item_id]] + 0.5
                        and cur_dist < min_dist
                    ):
                        min_dist = cur_dist
                        selected_item_id = item_id
                        selected_agent_anchor_idx = agent_anchor_idx
                if selected_item_id is None:
                    print("no attach")
                    return
                f = {"spring": 0.3, "rope": 0.1, "rod": 100}
                d = {"spring": 0, "rope": 0, "rod": 0.5}
                print("selected_item_id", selected_item_id)
                agent_size, object_size = (
                    SIZE[self.sizes[agent_id]],
                    SIZE[self.sizes[self.num_agents + selected_item_id]],
                )
                agent_anchors = [
                    (0, agent_size + object_size),
                    (0, -agent_size - object_size),
                    (agent_size + object_size, 0),
                    (-agent_size - object_size, 0),
                ]
                dfn = b2WeldJointDef(
                    # frequencyHz=f['rod'],
                    # dampingRatio=d['rod'],
                    bodyA=self.agents[agent_id],
                    bodyB=self.items[selected_item_id],
                    localAnchorA=agent_anchors[selected_agent_anchor_idx],
                    localAnchorB=(0, 0),
                )
                self.attached[agent_id] = self.world.CreateJoint(dfn)
                self.once_attached[agent_id] = True
                self.agents[agent_id].userData[-1] = selected_item_id
                if ("item", selected_item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][("item", selected_item_id)][1] = True
                self.items[selected_item_id].userData[-1] = agent_id
            return
        elif action == "detach":
            if self.attached[agent_id] is not None:
                self.world.DestroyJoint(self.attached[agent_id])
                self.attached[agent_id] = None
                item_id = self.agents[agent_id].userData[-1]
                self.items[item_id].userData[-1] = None
                if ("item", item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][("item", item_id)][1] = False
                self.agents[agent_id].userData[-1] = None
            return

        # print(agent_id, action)
        # if self.actions[agent_id] is None:
        #     self.actions[agent_id] = [action]
        # else:
        #     self.actions[agent_id].append(action)
        fx, fy = 0.0, 0.0
        df = STRENGTH[self.strengths[agent_id]] * self.NUM_STEPS_PER_TICK
        if action.startswith("turn"):
            df = max(df, STRENGTH[3] * self.NUM_STEPS_PER_TICK)
            size_factor = (
                SIZE[self.sizes[agent_id]] / 0.8
            ) ** 2  # size_factor=1.0 for size=0.8
            if action == "turnleft":
                # self.agents[agent_id].ApplyTorque(min(df * 1.0, 300), True)
                self.agents[agent_id].ApplyTorque(min(df, 300) * size_factor, True)
            else:
                # self.agents[agent_id].ApplyTorque(-min(df * 1.0, 300), True)
                self.agents[agent_id].ApplyTorque(
                    -min(df, 300) * size_factor, True
                )  # TODO for human game and in general the correct direction
                # self.agents[agent_id].ApplyTorque(min(df, 300) * size_factor, True) #for replay - old videos created with this bug
            return

        if action == "up":
            fy += df
        elif action == "down":
            fy -= df
        elif action == "left":
            fx -= df
        elif action == "right":
            fx += df
        elif action == "upleft":
            fx -= df * 0.707
            fy += df * 0.707
        elif action == "upright":
            fx += df * 0.707
            fy += df * 0.707
        elif action == "downleft":
            fx -= df * 0.707
            fy -= df * 0.707
        elif action == "downright":
            fx += df * 0.707
            fy -= df * 0.707
        elif action == "stop":
            # if self.strengths[agent_id] != 3:
            self.agents[agent_id].linearVelocity.x = 0
            self.agents[agent_id].linearVelocity.y = 0
            return
        elif action == "noforce":
            return
        else:
            print("ERROR: invalid action!")
        if action == "stop":
            print(fx, fy)
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(
            localPoint=(0.0, -SIZE[self.sizes[agent_id]] / 12.0)
        )
        self.agents[agent_id].ApplyForce(f, p, True)
        # self.agents[agent_id].ApplyLinearImpulse(f, p, True) #TODO human player

    def save_history(self):
        for agent_id in range(self.num_agents):
            history = copy.deepcopy(self.agents[agent_id].beliefs.particles)
            if self.full_obs[agent_id]:
                history = history[0]
            self.particle_history[agent_id].append(history)

    def belief_step(self, actions):
        # update belief
        for agent_id in range(self.num_agents):
            self.belief_update(agent_id, actions[agent_id])
            # print('updated particles',agent_id)
            # for i,ent in enumerate(self.agents[agent_id].beliefs):
            #     print('ent',i)
            #     for p in ent.particles:
            #         print(p)
        # never reach this part of code in MCTS_2agents!
        # #render gt & samples for each agent
        # if self.enable_renderer:
        #     self.render_FOV() #render gt as image (video in self.render)
        #     for agent_id in range(self.num_agents):
        #         curr_state = self.sample_belief_state(agent_id, actions[agent_id])
        #         self._display_imagined(agent_id, curr_state)
        #
        #         print('sample',agent_id)
        #         for ent in curr_state:
        #             print(ent)

    def step(self, replay=False, record_dir=None):
        """apply one step and update the environment"""
        # print("agent positions", self.agents_pos)
        self.steps += 1
        self.world.Step(self.TIME_STEP, 10, 10)
        # self.world.ClearForces()

        # print('vel:', round(self.agents[0].angularVelocity,2), round(self.agents[1].angularVelocity,2))
        self.agent_states = [_get_state_from_body(body) for body in self.agents]
        for agent_id, state in enumerate(self.agent_states):
            agent_pos, agent_vel, agent_angle = (
                state["pos"],
                state["vel"],
                state["angle"],
            )
            self.trajectories[agent_id].append(
                (agent_pos[0], agent_pos[1], agent_vel[0], agent_vel[1], agent_angle)
            )
            if self.actions[agent_id] is None:
                self.actions[agent_id] = [self.repeat_actions[agent_id]]
            else:
                self.actions[agent_id].append(self.repeat_actions[agent_id])
        self.item_states = [_get_state_from_body(body) for body in self.items]
        for item_id, state in enumerate(self.item_states):
            item_pos, item_vel, item_angle = state["pos"], state["vel"], state["angle"]
            self.trajectories[item_id + self.num_agents].append(
                (item_pos[0], item_pos[1], item_vel[0], item_vel[1], item_angle)
            )
        agent1beliefs_item1 = []  # TODO not general enough
        for t in range(self.NUM_STEPS_PER_TICK - 1):
            if not self.human_simulation:
                self.update_field_of_view()
                self.update_observations()
            if self.video and (
                self.clip_interval is None
                or self.steps > self.clip_interval[0]
                and self.steps <= self.clip_interval[1]
            ):
                if replay and self.flip_env is not None:
                    self.render_replay()
                else:
                    self.render()
            for agent_id in range(2):  # real action
                # if self.repeat_actions[agent_id] == 'stop' or t == self.NUM_STEPS_PER_TICK - 1:
                if self.repeat_actions[agent_id] not in ["attach", "detach"]:
                    self.send_action(agent_id, self.repeat_actions[agent_id])
            self.world.Step(self.TIME_STEP, 10, 10)
            if t == self.NUM_STEPS_PER_TICK - 2:
                self.world.ClearForces()
            # if self.repeat_actions[1] == 'stop':
            #     print('vel:', t, self.agents[1].linearVelocity.x, self.agents[1].linearVelocity.y)
            self.agent_states = [_get_state_from_body(body) for body in self.agents]
            for agent_id, state in enumerate(self.agent_states):
                agent_pos, agent_vel, agent_angle = (
                    state["pos"],
                    state["vel"],
                    state["angle"],
                )
                self.trajectories[agent_id].append(
                    (
                        agent_pos[0],
                        agent_pos[1],
                        agent_vel[0],
                        agent_vel[1],
                        agent_angle,
                    )
                )
            self.item_states = [_get_state_from_body(body) for body in self.items]
            for item_id, state in enumerate(self.item_states):
                item_pos, item_vel, item_angle = (
                    state["pos"],
                    state["vel"],
                    state["angle"],
                )
                self.trajectories[item_id + self.num_agents].append(
                    (item_pos[0], item_pos[1], item_vel[0], item_vel[1], item_angle)
                )
        # agent1beliefs_item1.append(belief_states[1][14:16])
        # for item_id in range(self.num_items):
        #     self.items[item_id].linearVelocity.x = 0
        #     self.items[item_id].linearVelocity.y = 0
        # self.items_vel = [_get_world_vel(item) for item in self.items]

        for agent_id in range(self.num_agents):
            self.agents[agent_id].angularVelocity = 0
            # human game
            if self.human_simulation:
                self.agents[agent_id].linearVelocity.x = 0
                self.agents[agent_id].linearVelocity.y = 0

        self.agent_states = [_get_state_from_body(body) for body in self.agents]

        self.running = not self.terminal()
        if not replay and not self.human_simulation:
            self.update_field_of_view()
            self.update_observations()
        if self.human_simulation:
            self.update_observations_human_simulation()
        if self.enable_renderer and (
            self.clip_interval is None
            or self.steps > self.clip_interval[0]
            and self.steps <= self.clip_interval[1]
        ):
            if replay and self.flip_env is not None:
                self.render_replay()
            else:
                self.render()
        return agent1beliefs_item1

    def get_obs(self, input_type):
        """get observation"""
        if input_type == "image":
            return (
                imresize(
                    _get_obs(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
                    (self.obs_dim[1], self.obs_dim[2]),
                )
                .transpose([2, 0, 1])
                .reshape((-1))
            )
        else:
            # FIXME: only for 2 agents
            agents_pos = [state["pos"] for state in self.agent_states]
            positions = [[x, y] for x, y in agents_pos]
            return [
                np.asarray(self.extract_state_feature(positions[0], positions[1]))
                for _ in range(2)
            ]

    # resample only entities that don't match FOV
    def sample_belief_state(self, agent_id, action):
        # assume sample_belief_state is called after belief update. particles should be feasible
        beliefs = self.agents[agent_id].beliefs
        curr_belief = copy.deepcopy(beliefs.curr_belief)
        if (
            curr_belief is None
        ):  # first call to this func, no prev state. choose random belief.
            next_belief = self.agents[agent_id].beliefs.sample_random_state()
        else:
            # transition curr belief
            # TODO expected_action - if observed should be gt, if estimated by level 1
            next_belief = self.transition(
                agent_id, curr_belief, action=action, expected_action=None
            )
            for obs in self.agents[agent_id].last_observations.keys():
                # currently observed.
                idx = int(obs[1])
                if obs[0] == "agent":
                    next_belief[idx]["pos"] = self.agent_states[idx]["pos"]
                    next_belief[idx]["vel"] = self.agent_states[idx]["vel"]
                    next_belief[idx]["angle"] = self.agent_states[idx]["angle"]
                    next_belief[idx]["angleVel"] = self.agent_states[idx]["angleVel"]
                    next_belief[idx]["attached"] = self.agent_states[idx]["attached"]
                else:
                    next_belief[self.num_agents + idx]["pos"] = self.item_states[idx][
                        "pos"
                    ]
                    next_belief[self.num_agents + idx]["vel"] = self.item_states[idx][
                        "vel"
                    ]
                    next_belief[self.num_agents + idx]["angle"] = self.item_states[idx][
                        "angle"
                    ]
                    next_belief[self.num_agents + idx]["angleVel"] = self.item_states[
                        idx
                    ]["angleVel"]
                    next_belief[self.num_agents + idx]["attached"] = self.item_states[
                        idx
                    ]["attached"]

            matches = self.belief_matches_observation(agent_id, next_belief)
            # if matches obs: return curr sample transitioned
            if not matches[0]:  # for each entity that does not match
                # TODO is len(matches[1])>1, entities are never updated. need to be updated simultaneously.
                for entity in matches[1]:
                    if (
                        entity == ("agent", agent_id)
                        or entity in self.agents[agent_id].last_observations
                    ):
                        raise TypeError("gt does not match")  # set earlier.
                    else:
                        print("updating belief ", entity, " agent", agent_id)
                        entity_type, entity_id = entity
                        if entity_type == "item":
                            entity_id += self.num_agents
                        # TODO sample based on frequency of grid pos out of FOV, for loop
                        entity_particles = [
                            state[entity_id] for state in beliefs.particles
                        ]
                        entity_grid_pos = [
                            self.world_point_to_grid_cell(p["pos"][0], p["pos"][1])
                            for p in entity_particles
                        ]
                        # unique_pos has grid cells
                        unique_pos, inv_idx, cnt = np.unique(
                            entity_grid_pos,
                            return_inverse=True,
                            return_counts=True,
                            axis=0,
                        )
                        idx = [
                            s[1]
                            for s in sorted(
                                zip(cnt, list(range(len(unique_pos)))), reverse=True
                            )
                        ]
                        particles_by_freq = []  # indices 1-->n_particles by pos freq
                        for l in [(list(np.where(inv_idx == i)[0])) for i in idx]:
                            particles_by_freq += l
                        for p_idx in particles_by_freq:
                            next_resampled = copy.deepcopy(next_belief)
                            next_resampled[entity_id] = copy.deepcopy(
                                beliefs.particles[int(p_idx)][entity_id]
                            )
                            # next_resampled = self.valid_belief(agent_id, next_resampled)
                            if self.belief_matches_observation(
                                agent_id, next_resampled
                            )[0]:
                                next_belief = next_resampled
                                break  # for loop
                    # matches = self.belief_matches_observation(agent_id, next_belief)

        self.agents[agent_id].beliefs.curr_belief = copy.deepcopy(next_belief)
        return copy.deepcopy(next_belief)

    def sample_random_belief_state(self, agent_id):
        beliefs = self.agents[agent_id].beliefs
        particle_idx = random.choice(list(range(len(beliefs))))
        next_belief = beliefs.sample_state(idx=particle_idx)
        self.agents[agent_id].curr_belief = next_belief
        return copy.deepcopy(next_belief)

    def get_belief_certainty(self, agent_id):
        """score for each entity except agent.
        calculated as the average belief certainty over the entity perimeter."""
        self._setup_tmp(self.env_id, self.agents[agent_id].beliefs.curr_belief)
        scores = []
        for other_agent_id in range(self.num_agents):
            if other_agent_id == agent_id:
                continue
            grid_cells = self.get_perimeter_grid_cells(
                "agent", other_agent_id, self.tmp_agents[other_agent_id]
            )
            scores.append(
                np.average(
                    [self.agents[agent_id].certainty[c[0], c[1]] for c in grid_cells]
                )
            )
        for item_id in range(self.num_items):
            grid_cells = self.get_perimeter_grid_cells(
                "item", item_id, self.tmp_item_states[item_id]
            )
            item_score = np.average(
                [self.agents[agent_id].certainty[c[0], c[1]] for c in grid_cells]
            )
            scores.append(item_score)
        return scores

    def get_state(self, agent_id=None):
        """
        get the state of the world
        """
        certainty = [agent.certainty.copy() for agent in self.agents]
        if agent_id is None:
            return [_get_state_from_body(agent) for agent in self.agents] + [
                _get_state_from_body(item) for item in self.items
            ]
        else:
            return [_get_state_from_body(agent) for agent in self.agents] + [
                _get_state_from_body(item) for item in self.items
            ], [
                self.agents[i].certainty.copy()
                if i == agent_id or self.full_obs[agent_id]
                else np.zeros(self.room_dim)
                for i in range(self.num_agents)
            ]

    def terminal(self):
        """check if the goal is achieved"""
        return False

    def get_reward(self):
        dist_to_bottom_left_corner = _get_dist(
            self.item_states[0]["pos"], (16 - 10 + 2, 12 - 10 + 2)
        )
        normalized_dist = dist_to_bottom_left_corner / _get_dist(
            (16, 12 + 4), (16 - 10 + 2, 12 - 10 + 2)
        )
        # r = [-normalized_dist - 0.2, -normalized_dist - 0.2]
        r = [1.0 - normalized_dist, 1.0 - normalized_dist]
        return r[self.planning_agent]

    def get_plan_score(
        self, sim_envs, agent_id, method, expected_pos=None, simulation=None, T=5
    ):
        """agent_id observing agent. simulation - action_probs, actual_simulation_action_idxs, world_states"""
        if method == "action_prob":  # similar actions --> high score
            if self.steps < T:
                self.trajectory_prob = 1
            elif self.steps % T == 0:
                # actual action idxs taken, simulated action probs
                print("get_plan_score")
                print(simulation["action_probs"])
                print(sim_envs[1 - agent_id].action_idxs[1 - agent_id])
                self.trajectory_prob = np.prod(
                    [
                        simulation["action_probs"][t][
                            sim_envs[1 - agent_id].action_idxs[1 - agent_id][t]
                        ]
                        for t in range(self.steps - T, self.steps)
                    ]
                )
                # self.trajectory_prob = np.prod([sim_envs[agent_id].action_probs[1-agent_id][t]\
                #                                [sim_envs[1-agent_id].action_idxs[1-agent_id][t]] \
                #                            for t in range(self.steps-T,self.steps)])
                # #calculate and update prior based on particle history for other agent's target item
                # entity_id = GOAL[self.goals[1-agent_id]][2]
                # gt_states = self.particle_history[agent_id]
                # particles = self.particle_history[1-agent_id]
                # self.belief_prior = 0
                # n_particles = len(self.particle_history[0][0])
                # for t in range(self.steps,self.steps+T):
                #     self.belief_prior += np.sum([self.equivalent_states(gt_states[t][entity_id],particle[entity_id]) \
                #                                     for particle in particles[t]])
                # self.belief_prior /= T * n_particles
            plan_score = self.trajectory_prob * self.belief_prior
        elif method == "pos_disparity":  # close pos --> low score
            true_pos = self.agent_states[1 - agent_id]["pos"]
            # expected_pos = simulation['world_states'][self.steps-1][0][1-agent_id]['pos']
            print("pos_disparity", expected_pos, true_pos)
            plan_score = math.hypot(
                true_pos[0] - expected_pos[0], true_pos[1] - expected_pos[1]
            )
        elif method == "visibility":  # visible --> high score
            plan_score = (
                1.0
                if sim_envs[1 - agent_id].in_field_of_view(
                    1 - agent_id, ("agent", agent_id)
                )
                else 0.0
            )
        return plan_score

    def _get_dist_room(self, pos, room_id):
        cur_room_id = self._get_room_id(pos)
        if cur_room_id != room_id:
            door_id = self.connected[cur_room_id][room_id]
            # if door_id is not None:
            #     print(pos, cur_room_id, room_id)
            if door_id is None:
                return self._max_dist
            # print(self.doors_pos[door_id])
            return _get_dist(pos, self.doors_pos[door_id]) + _get_dist(
                self.doors_pos[door_id], self.landmark_corners[room_id]
            )
        else:
            # print(pos, cur_room_id, room_id)
            return _get_dist(pos, self.landmark_corners[room_id])

    def _get_dist_room_point(self, pos, room_id):
        cur_room_id = self._get_room_id(pos)
        if cur_room_id != room_id:
            door_id = self.connected[cur_room_id][room_id]
            if door_id is None:
                return None
            return self.doors_pos[door_id]
        else:
            return None  # self.landmark_corners[room_id]

    def _get_doors(self, start_room_id, path):
        cur_room_id = start_room_id
        doors = []
        for target_room_id in path:
            door_id = self.connected[cur_room_id][target_room_id]
            doors.append(door_id)
            cur_room_id = target_room_id
        return doors

    def _get_dist_pos(self, pos1, pos2):
        if self.env_id == 0:
            return _get_dist(pos1, pos2)
        room_id1 = self._get_room_id(pos1)
        room_id2 = self._get_room_id(pos2)
        if room_id1 != room_id2:
            path = self.path[(room_id1, room_id2)]
            if path is None:
                return self._max_dist * 10
            doors = self.doors[(room_id1, room_id2)]
            dist = _get_dist(pos1, self.doors_pos[doors[0]]) + _get_dist(
                pos2, self.doors_pos[doors[-1]]
            )
            if len(doors) > 1:
                for door_id in range(0, len(doors) - 1):
                    dist += _get_dist(
                        self.doors_pos[doors[door_id]],
                        self.doors_pos[doors[door_id + 1]],
                    )
        else:
            dist = _get_dist(pos1, pos2)
        return dist

    def is_far(self):
        return (
            self._get_dist_pos(self.agent_states[0]["pos"], self.agent_states[1]["pos"])
            > 14.14
        )

    def get_reward_state(
        self,
        agent_id,
        curr_state,
        action,
        t=0,
        T=0,
        goal=None,
        prev_certainty=None,
        curr_certainty=None,
    ):
        agents_pos = [agent["pos"] for agent in curr_state[: self.num_agents]]
        items_pos = [item["pos"] for item in curr_state[self.num_agents :]]
        attached = [agent["attached"] for agent in curr_state[: self.num_agents]]

        if goal is not None and len(goal) == 2:
            # 2 goals
            normalized_dist = 0
            ws = [0.5, 0.5]
            for g, w in zip(goal, ws):
                dist_to_goal = self._get_dist_pos(
                    items_pos[g[1]], self.landmark_centers[g[2]]
                )
                normalized_dist += w * g[3] * dist_to_goal / self._max_dist
            normalized_dist /= len(goal)
        else:
            # next_state = self.transition(curr_state, action)
            # print(next_state)
            goal = self.goals[agent_id] if goal is None else goal
            if goal[0] == "stop":
                return 0.0 if action == "noforce" else -COST[self.costs[agent_id]]

            # dA = 2 * math.pi / len(self.viz_map)

            """TODO: RA & RO"""
            if goal[0] == "LMA":
                dist_to_goal = self._get_dist_pos(
                    agents_pos[goal[1]], self.landmark_centers[goal[2]]
                )
            elif goal[0] == "RA":
                dist_to_goal = self._get_dist_room(agents_pos[goal[1]], goal[2])
                # p1_cell = self.world_point_to_grid_cell(agents_pos[goal[1]][0], agents_pos[goal[1]][1])
                # p2 = self._get_dist_room_point(agents_pos[goal[1]], goal[2])
                # if p2 is not None:
                #     p2_cell = self.world_point_to_grid_cell(p2[0], p2[1])
                #     angle = math.atan2(math.sin(curr_state[goal[1]]['angle'] + math.pi * 0.5), math.cos(curr_state[goal[1]]['angle'] + math.pi * 0.5))
                #     a = int((angle + math.pi) / dA)
                #     if self.viz_map[a,p1_cell[0],p1_cell[1],p2_cell[0],p2_cell[1]] > 0:
                #         dist_to_goal -= self._max_dist * 0.2
                # print(agents_pos[goal[1]], math.degrees(angle), p2, a, p1_cell, p2_cell, self.viz_map[a,p1_cell[0],p1_cell[1],p2_cell[0],p2_cell[1]])
                # if agent_id == 0:
                #     print(agents_pos[goal[1]], goal[2], dist_to_goal)
            elif goal[0] == "LMO" or goal[0] == "LMOF":
                dist_to_goal = self._get_dist_pos(
                    items_pos[goal[1]], self.landmark_centers[goal[2]]
                )
            elif goal[0] == "RO":
                dist_to_goal = self._get_dist_room(items_pos[goal[1]], goal[2])
                room_id = self._get_room_id(items_pos[goal[1]])
                if room_id == goal[2]:
                    dist_to_goal -= self._max_dist * 10
                # else:
                #     doors = self.doors[(room_id, goal[2])]
                #     # print(self._is_blocked(doors, SIZE[self.sizes[agent_id]]), self._get_dist_room(agents_pos[agent_id], goal[2]), dist_to_goal)
                #     if self._is_blocked(doors, SIZE[self.sizes[agent_id]] * 1.5) and self._get_dist_room(agents_pos[agent_id], goal[2]) < dist_to_goal:
                #         dist_to_goal += (dist_to_goal - self._get_dist_room(agents_pos[agent_id], goal[2]))
            elif goal[0] == "LMOP":
                dist_to_goal = self._get_dist_pos(
                    items_pos[goal[1]], self.landmark_centers[goal[2]]
                ) - 0.5 * self._get_dist_pos(
                    items_pos[goal[1]], agents_pos[1 - agent_id]
                )
            elif goal[0] == "TE":
                if goal[2] < self.num_agents:
                    dist_to_goal = self._get_dist_pos(
                        agents_pos[goal[1]], agents_pos[goal[2]]
                    )
                    p2 = agents_pos[goal[2]]
                else:
                    dist_to_goal = self._get_dist_pos(
                        agents_pos[goal[1]], items_pos[goal[2] - self.num_agents]
                    )
                    p2 = items_pos[goal[2] - self.num_agents]
                # p2_cell = self.world_point_to_grid_cell(p2[0], p2[1])
                # p1_cell = self.world_point_to_grid_cell(agents_pos[goal[1]][0], agents_pos[goal[1]][1])
                # angle = math.atan2(math.sin(curr_state[goal[1]]['angle'] + math.pi * 0.5), math.cos(curr_state[goal[1]]['angle'] + math.pi * 0.5))
                # a = int((angle + math.pi) / dA)
                # if self.viz_map[a,p1_cell[0],p1_cell[1],p2_cell[0],p2_cell[1]] > 0:
                #     dist_to_goal -= self._max_dist * 0.2
                # print(agents_pos[goal[1]], math.degrees(angle), p2, a, p1_cell, p2_cell, self.viz_map[a,p1_cell[0],p1_cell[1],p2_cell[0],p2_cell[1]])
            elif goal[0] == "GE":
                """TODO: more objects"""
                dist_to_goal = (
                    0.0 if attached[goal[1]] == goal[2] - self.num_agents else 1.0
                )
            elif goal[0] == "TEV":
                if goal[2] < self.num_agents:
                    dist_to_goal = self._get_dist_pos(
                        agents_pos[goal[1]], agents_pos[goal[2]]
                    )
                else:
                    dist_to_goal = self._get_dist_pos(
                        agents_pos[goal[1]], items_pos[goal[2] - self.num_agents]
                    )
                opponent_id = 1 - goal[1]  # TODO: only works for 2 agents
                visible = self.in_field_of_view(opponent_id, ("agent", agent_id))
                if visible:
                    dist_to_goal += self._max_dist
            elif goal[0] == "OBS":
                dist_to_goal = (
                    0.0 if self.in_field_of_view(agent_id, ("agent", goal[2])) else 1.0
                )
            elif goal[0] == "V":
                dist_to_goal = -self._get_dist_pos(
                    agents_pos[goal[1]], agents_pos[goal[2]]
                )
                dist_to_goal += (
                    self._max_dist
                    if self.in_field_of_view(goal[2], ("agent", goal[1]))
                    else 0.0
                )
            elif goal[0] == "PA":
                dist_to_goal = self._get_dist_pos(agents_pos[goal[1]], goal[2])
                dist_to_goal += (
                    self._max_dist
                    if self.in_field_of_view(1 - goal[1], ("agent", goal[1]))
                    else 0.0
                )
            else:
                raise ValueError("Invalid goal!")
            normalized_dist = goal[3] * dist_to_goal / self._max_dist
        # if goal[0] == 'LMA':
        #     normalized_dist = goal[3] * dist_to_goal / self._max_dist# _get_dist_lm(self.init_agents_pos[goal[1]], self.landmark_centers[goal[2]])
        # else:
        #     normalized_dist = goal[3] * dist_to_goal / self._max_dist #_get_dist(self.init_items_pos[goal[1]], self.landmark_centers[goal[2]])
        # r = [-normalized_dist - 0.2, -normalized_dist - 0.2]
        # print(agent_id, items_pos, self.landmark_centers[goal[2]], dist_to_goal, normalized_dist)
        # # input('press any key to continue...')
        r = -normalized_dist
        cost = 0.0 if action == "noforce" else COST[self.costs[agent_id]]
        if action in [
            "down",
            "left",
            "right",
            "upleft",
            "upright",
            "downleft",
            "downright",
        ]:
            cost += 0.02
        r -= cost
        # if prev_certainty is not None and curr_certainty is not None:
        #     r += 1.0 * np.sum(curr_certainty[agent_id] - prev_certainty[agent_id])

        return r

    def is_terminal(self, curr_state, t):
        """check if the goal is achieved"""
        # if t == self.max_episode_length: return True
        # ITEMBASE = 8
        # items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1])]
        # items_vel = [(curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        # dist_to_bottom_left_corner = _get_dist(items_pos[0], (16 - 10 + 2, 12 - 10 + 2))
        # goal_dist = dist_to_bottom_left_corner
        # d_threshold = 0.5
        # v_threshold = 0.01
        # return _norm(items_vel[0]) < v_threshold and goal_dist < d_threshold
        return False

    def _setup_tmp(self, env_id, curr_state, curr_certainty=None):
        """setup a new espisode from beliefs. some vars used for construction are used from true env."""

        agents_pos = [agent["pos"] for agent in curr_state[: self.num_agents]]
        items_pos = [item["pos"] for item in curr_state[self.num_agents :]]
        agents_vel = [agent["vel"] for agent in curr_state[: self.num_agents]]
        agents_angelVel = [agent["angleVel"] for agent in curr_state[: self.num_agents]]
        items_vel = [item["vel"] for item in curr_state[self.num_agents :]]
        agents_angle = [agent["angle"] for agent in curr_state[: self.num_agents]]
        attached = [agent["attached"] for agent in curr_state[: self.num_agents]]

        if self.flip_env is not None:
            agents_pos = flip_positions(agents_pos, self.flip_env)
            items_pos = flip_positions(items_pos, self.flip_env)
            agents_vel = flip_positions(agents_vel, self.flip_env)
            agents_angle = flip_angles(agents_angle, self.flip_env)

        if curr_certainty is None:
            curr_certainty = [np.zeros(self.room_dim) for _ in range(self.num_agents)]

        # self.tmp_world = world(contactListener=MyContactListener(), gravity=(0, 0), doSleep=True)
        self.tmp_world = world(gravity=(0, 0), doSleep=True)
        self.tmp_room = self.tmp_world.CreateBody(position=(16, 12))
        self.tmp_room.CreateEdgeChain(
            [
                (-self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, self.room_dim[1] / 2 - self.door_length),
            ]
        )
        if self.enable_renderer:
            self.tmp_screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32
            )
            pygame.display.set_caption("Maze_v1 belief")
            self.tmp_clock = pygame.time.Clock()
        # build maze
        self.tmp_wall_segs = []
        env_def = self.maze_sampler.get_env_def(env_id)
        maze_def = env_def["maze_def"]
        if self.flip_env is not None:
            maze_def = flip_maze(maze_def, self.flip_env)
        for wall_id, wall in enumerate(maze_def):
            if WALL_SEGS[wall] is not None:
                seg = [WALL_SEGS[wall][0], WALL_SEGS[wall][1]]
                if wall_id == 0:
                    # if env_id == 16:
                    #     seg[1] -= 3
                    self.tmp_room.CreateEdgeChain([(0, seg[0]), (0, seg[1])])
                    self.tmp_wall_segs.append([(16, seg[0] + 12), (16, seg[1] + 12)])
                elif wall_id == 1:
                    self.tmp_room.CreateEdgeChain([(-seg[0], 0), (-seg[1], 0)])
                    self.tmp_wall_segs.append([(16 - seg[0], 12), (16 - seg[1], 12)])
                elif wall_id == 2:
                    self.tmp_room.CreateEdgeChain([(0, -seg[0]), (0, -seg[1])])
                    self.tmp_wall_segs.append([(16, -seg[0] + 12), (16, -seg[1] + 12)])
                elif wall_id == 3:
                    self.tmp_room.CreateEdgeChain([(seg[0], 0), (seg[1], 0)])
                    self.tmp_wall_segs.append([(16 + seg[0], 12), (16 + seg[1], 12)])

        self.tmp_agents, self.tmp_items = [], []
        # self.tmp_trajectories = [None] * (self.num_agents + self.num_items + 1)
        # add agents
        for agent_id in range(self.num_agents):
            R = SIZE[self.sizes[agent_id]]
            x, y = agents_pos[agent_id]
            body = self.tmp_world.CreateDynamicBody(
                position=(x, y),
                angle=agents_angle[agent_id],
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(
                        vertices=[
                            (-1.5 * R, -R),
                            (-0.5 * R, R),
                            (0.5 * R, R),
                            (1.5 * R, -R),
                        ]
                    ),
                    density=DENSITY[self.densities[agent_id]],
                    friction=FRICTION,
                ),
            )
            body.linearVelocity.x = agents_vel[agent_id][0]
            body.linearVelocity.y = agents_vel[agent_id][1]
            body.angularVelocity = agents_angelVel[agent_id]
            # TODO should these be simulated in some way as well? (userData for sampling close entities --> collision)
            body.userData = [
                ("agent", agent_id),
                self.touch_sensor,
                None,
            ]  # 3rd entry for attached item_id (None if not attached)
            body.field_of_view = np.zeros(self.room_dim)
            body.certainty = curr_certainty[agent_id].copy()
            # body.last_observations = {}
            self.tmp_agents.append(body)
            self.tmp_repeat_actions = [None] * 2
            self.tmp_actions = [None] * self.num_agents
            # TODO is this needed?
            # self.tmp_trajectories[agent_id] = [(x, y, 0, 0)]
        self.tmp_agent_states = [_get_state_from_body(body) for body in self.tmp_agents]
        for agent_id, state in enumerate(self.tmp_agent_states):
            agent_pos, agent_vel, agent_angle = (
                state["pos"],
                state["vel"],
                state["angle"],
            )
            self.tmp_trajectories[agent_id].append(
                (agent_pos[0], agent_pos[1], agent_vel[0], agent_vel[1], agent_angle)
            )

        # add items
        ITEM_BASE = self.num_agents
        for item_id in range(self.num_items):
            index = ITEM_BASE + item_id
            x, y = items_pos[item_id]
            body = self.tmp_world.CreateDynamicBody(position=(x, y))
            body.CreateCircleFixture(
                radius=SIZE[self.sizes[index]],
                density=DENSITY[self.densities[index]],
                friction=FRICTION,
                restitution=0,
            )
            body.linearVelocity.x = items_vel[item_id][0]
            body.linearVelocity.y = items_vel[item_id][1]
            body.userData = [
                ("item", item_id),
                None,
            ]  # 2nd entry for attached agent_id (None if not attached)
            self.tmp_items.append(body)
            # self.trajectories[index] = [(x, y, 0, 0)]
        self.tmp_item_states = [_get_state_from_body(body) for body in self.tmp_items]
        for item_id, state in enumerate(self.tmp_item_states):
            item_pos, item_vel, item_angle = state["pos"], state["vel"], state["angle"]
            self.tmp_trajectories[item_id + self.num_agents].append(
                (item_pos[0], item_pos[1], item_vel[0], item_vel[1], item_angle)
            )

        # friction
        self.tmp_groundBody = self.tmp_world.CreateStaticBody(
            position=(16, 12),
            shapes=polygonShape(box=(self.room_dim[0], self.room_dim[1])),
        )
        for body in self.tmp_agents + self.tmp_items:
            dfn = b2FrictionJointDef(
                bodyA=body,
                bodyB=self.tmp_groundBody,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                maxForce=100,
                maxTorque=0,
            )
            self.tmp_world.CreateJoint(dfn)

        self.tmp_attached = [None] * self.num_agents
        # self.once_attached = [False] * self.num_agents
        """TODO more items"""
        for agent_id in range(self.num_agents):
            if attached[agent_id] is not None:
                # print(self.agents[agent_id].userData[2])
                f = {"spring": 0.3, "rope": 0.1, "rod": 100}
                d = {"spring": 0, "rope": 0, "rod": 0.5}
                dfn = b2DistanceJointDef(
                    frequencyHz=f["rod"],
                    dampingRatio=d["rod"],
                    bodyA=self.tmp_agents[agent_id],
                    bodyB=self.tmp_items[
                        attached[agent_id]
                    ],  # TODO: need item id!!! #self.tmp_items[0],
                    localAnchorA=(0, 0),
                    localAnchorB=(0, 0),
                )
                self.tmp_attached[agent_id] = self.tmp_world.CreateJoint(dfn)
                self.tmp_agents[agent_id].userData[-1] = attached[
                    agent_id
                ]  # TODO: need item id!!!
                self.tmp_items[item_id].userData[-1] = agent_id

    def _send_action_tmp(self, agent_id, action):
        if action is None:
            return
        self.tmp_repeat_actions[agent_id] = action
        if action not in (
            self.action_space[: -self.action_space_types[agent_id]]
            if self.action_space_types[agent_id]
            else self.action_space
        ):
            return

        if action == "attach":
            head_mid_point = self.tmp_agents[agent_id].GetWorldPoint(
                localPoint=(0.0, SIZE[self.sizes[agent_id]])
            )
            tail_mid_point = self.tmp_agents[agent_id].GetWorldPoint(
                localPoint=(0.0, -SIZE[self.sizes[agent_id]])
            )
            right_mid_point = self.tmp_agents[agent_id].GetWorldPoint(
                localPoint=(SIZE[self.sizes[agent_id]], 0.0)
            )
            left_mid_point = self.tmp_agents[agent_id].GetWorldPoint(
                localPoint=(-SIZE[self.sizes[agent_id]], 0.0)
            )
            all_agent_pos = [
                head_mid_point,
                tail_mid_point,
                right_mid_point,
                left_mid_point,
            ]
            # print(all_agent_pos)
            if self.tmp_attached[agent_id] is None:
                min_dist = 1e6
                selected_item_id = None
                selected_agent_anchor_idx = None
                for item_id in range(self.num_items):
                    all_dist = [
                        _get_dist(pos, self.item_states[item_id]["pos"])
                        for pos in all_agent_pos
                    ]
                    cur_dist = min(all_dist)
                    agent_anchor_idx = all_agent_pos.index(
                        all_agent_pos[all_dist.index(cur_dist)]
                    )
                    # print(cur_dist, SIZE[self.sizes[self.num_agents + item_id]] + 0.2)
                    if (
                        cur_dist < SIZE[self.sizes[self.num_agents + item_id]] + 0.5
                        and cur_dist < min_dist
                    ):
                        min_dist = cur_dist
                        selected_item_id = item_id
                        selected_agent_anchor_idx = agent_anchor_idx
                if selected_item_id is None:
                    # print('no attach')
                    return
                f = {"spring": 0.3, "rope": 0.1, "rod": 100}
                d = {"spring": 0, "rope": 0, "rod": 0.5}
                # print(selected_item_id)
                agent_size, object_size = (
                    SIZE[self.sizes[agent_id]],
                    SIZE[self.sizes[self.num_agents + selected_item_id]],
                )
                agent_anchors = [
                    (0, agent_size + object_size),
                    (0, -agent_size - object_size),
                    (agent_size + object_size, 0),
                    (-agent_size - object_size, 0),
                ]
                dfn = b2WeldJointDef(
                    # frequencyHz=f['rod'],
                    # dampingRatio=d['rod'],
                    bodyA=self.tmp_agents[agent_id],
                    bodyB=self.tmp_items[selected_item_id],
                    localAnchorA=agent_anchors[selected_agent_anchor_idx],
                    localAnchorB=(0, 0),
                )
                self.tmp_attached[agent_id] = self.tmp_world.CreateJoint(dfn)
                # self.once_attached[agent_id] = True
                # print(self.tmp_agents[agent_id].userData)
                # print(selected_item_id)
                self.tmp_agents[agent_id].userData[-1] = selected_item_id
                """TODO: touch sensor"""
                if ("item", selected_item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][("item", selected_item_id)][1] = True
                self.tmp_items[selected_item_id].userData[-1] = agent_id
            return
        elif action == "detach":
            if self.tmp_attached[agent_id] is not None:
                self.tmp_world.DestroyJoint(self.tmp_attached[agent_id])
                self.tmp_attached[agent_id] = None
                item_id = self.tmp_agents[agent_id].userData[2]
                self.tmp_items[item_id].userData[-1] = None
                if ("item", item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][("item", item_id)][1] = False
                self.tmp_agents[agent_id].userData[-1] = None
            return

        fx, fy = 0.0, 0.0
        df = STRENGTH[self.strengths[agent_id]] * self.NUM_STEPS_PER_TICK

        if action.startswith("turn"):
            df = max(df, STRENGTH[3] * self.NUM_STEPS_PER_TICK)
            size_factor = (
                SIZE[self.sizes[agent_id]] / 0.8
            ) ** 2  # size_factor=1.0 for size=0.8
            if action == "turnleft":
                self.tmp_agents[agent_id].ApplyTorque(min(df, 300) * size_factor, True)
            else:
                self.tmp_agents[agent_id].ApplyTorque(-min(df, 300) * size_factor, True)
            return

        if action == "up":
            fy += df
        elif action == "down":
            fy -= df
        elif action == "left":
            fx -= df
        elif action == "right":
            fx += df
        elif action == "upleft":
            fx -= df * 0.707
            fy += df * 0.707
        elif action == "upright":
            fx += df * 0.707
            fy += df * 0.707
        elif action == "downleft":
            fx -= df * 0.707
            fy -= df * 0.707
        elif action == "downright":
            fx += df * 0.707
            fy -= df * 0.707
        elif action == "stop":
            # if self.strengths[agent_id] != 3:
            self.tmp_agents[agent_id].linearVelocity.x = 0
            self.tmp_agents[agent_id].linearVelocity.y = 0
            return
        elif action == "noforce":
            return
        else:
            print("ERROR: invalid action!")
        if action == "stop":
            print(fx, fy)
        f = self.tmp_agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.tmp_agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.tmp_agents[agent_id].ApplyForce(f, p, True)
        # self.tmp_agents[agent_id].ApplyLinearImpulse(f, p, True)

    # def _step_valid(self):
    #     """based on _step_tmp, move only to eliminate implausible collisions/overlap"""
    #     # self.tmp_steps += 1
    #     self.tmp_world.Step(self.TIME_STEP, 10, 10)
    #     for t in range(self.NUM_STEPS_PER_TICK - 1):
    #         self.tmp_world.Step(self.TIME_STEP, 10, 10)
    #         if t == self.NUM_STEPS_PER_TICK - 2:
    #             self.tmp_world.ClearForces()

    #     for agent_id in range(self.num_agents):
    #         self.tmp_agents[agent_id].angularVelocity = 0
    #     self.tmp_agent_states = [_get_state_from_body(body) for body in self.tmp_agents]
    #     self.tmp_item_states = [_get_state_from_body(body) for body in self.tmp_items]
    #     self.tmp_running = not self.terminal()

    def _step_tmp(self, update_certainty=True):
        """apply one step and update the environment"""
        # self.tmp_steps += 1
        self.tmp_world.Step(self.TIME_STEP, 10, 10)
        # self.tmp_world.ClearForces()
        for agent_id in range(self.num_agents):
            if self.tmp_actions[agent_id] is None:
                self.tmp_actions[agent_id] = [self.tmp_repeat_actions[agent_id]]
            else:
                self.tmp_actions[agent_id].append(self.tmp_repeat_actions[agent_id])

        for t in range(self.NUM_STEPS_PER_TICK - 1):
            """TODO: also update these intermediate steps?"""
            # if update_certainty:
            #     self._update_field_of_view_tmp()
            for agent_id in range(2):
                # if self.repeat_actions[agent_id] == 'stop' or t == self.NUM_STEPS_PER_TICK - 1:
                if self.tmp_repeat_actions[agent_id] not in ["attach", "detach"]:
                    self._send_action_tmp(agent_id, self.tmp_repeat_actions[agent_id])
            self.tmp_world.Step(self.TIME_STEP, 10, 10)
            if t == self.NUM_STEPS_PER_TICK - 2:
                self.tmp_world.ClearForces()

        for agent_id in range(self.num_agents):
            self.tmp_agents[agent_id].angularVelocity = 0
        self.tmp_agent_states = [_get_state_from_body(body) for body in self.tmp_agents]
        self.tmp_item_states = [_get_state_from_body(body) for body in self.tmp_items]

        if update_certainty:
            self._update_field_of_view_tmp()
        self.tmp_running = not self.terminal()

    def _get_state_tmp(self, agent_id=None):
        """
        get the imagined state
        """
        certainty = [agent.certainty.copy() for agent in self.tmp_agents]
        if agent_id is None:
            return [_get_state_from_body(agent) for agent in self.tmp_agents] + [
                _get_state_from_body(item) for item in self.tmp_items
            ]
        else:
            return [_get_state_from_body(agent) for agent in self.tmp_agents] + [
                _get_state_from_body(item) for item in self.tmp_items
            ], [
                self.tmp_agents[i].certainty.copy()
                if i == agent_id or self.full_obs[agent_id]
                else np.zeros(self.room_dim)
                for i in range(self.num_agents)
            ]

    def _display_imagined(
        self, agent_id, curr_particles, record_dir="", path_extension=""
    ):
        """based on transition func (simulate one step)"""
        """curr_state = believed state"""

        # self._setup_tmp(self.env_id, curr_state)
        # self.belief_render(agent_id, record_dir, path_extension)

        self.belief_render_all_particles(
            agent_id, curr_particles, record_dir, path_extension
        )
        if self.render_separate_particles:
            self.belief_render_all_particles_separately(
                agent_id, curr_particles, record_dir, path_extension
            )

    # def valid_belief(self, agent_id, curr_state, action=None):
    #     """simulate one step with 0 velocity to make sure belief is valid"""
    #     #to eliminate movement
    #     # curr_agents_vel, curr_items_vel = np.zeros_like(curr_agents_vel), np.zeros_like(curr_items_vel)
    #     curr_state_no_movement = copy.deepcopy(curr_state)
    #     for entity_id,entity in enumerate(curr_state_no_movement):
    #         entity['vel'] = np.zeros_like(entity['vel'])

    #     self._setup_tmp(self.env_id, curr_state_no_movement)
    #     # print('_tmp_transition ', agent_id)
    #     #degenerate step, just to make sure valid belief locations.
    #     self._step_valid()
    #     next_state = self._get_state_tmp()

    #     for entity_id,entity in enumerate(curr_state):
    #         next_state[entity_id]['vel'] = entity['vel']

    #     return next_state

    def equivalent_states(self, state1, state2):
        # TODO note precision when comparing. (after the dot / can compare distance to thres)
        # TDOO compare not only pos.
        # if state1['pos'][0] == state2['pos'][0] and state1['pos'][1] == state2['pos'][1] \
        #     and state1['angle'] == state2['angle'] \
        #     and state1['vel'][0] == state2['vel'][0] and state1['vel'][1] == state2['vel'][1] \
        #     and state1['angleVel'][2] == state2['angleVel'][2]:
        #         return True

        return (
            abs(round(state1["pos"][0], 2) - round(state2["pos"][0], 2)) <= 1
            and abs(round(state1["pos"][1], 2) - round(state2["pos"][1], 2)) <= 1
        )

    def belief_matches_observation(self, agent_id, entity_id, state):
        """check that a given entity's state matches observations"""
        res = True
        # tmp_agent pos&angle (or FOV?) match the true agent's one.
        if agent_id == entity_id:
            if not self.equivalent_states(self.agent_states[agent_id], state[agent_id]):
                # return False, ('self',agent_id)
                res = False
            return res

        # each believed entity (vertices) that falls into the FOV --> check True
        self._setup_tmp(self.env_id, state)
        if entity_id < self.num_agents:
            agent_grid_cells = self.get_perimeter_grid_cells(
                "agent", entity_id, self.tmp_agents[entity_id]
            )
            if (
                sum(
                    [
                        grid_cell[0] < 0
                        or grid_cell[0] > self.room_dim[0]
                        or grid_cell[1] < 0
                        or grid_cell[1] > self.room_dim[1]
                        for grid_cell in agent_grid_cells
                    ]
                )
                > 0
            ):
                res = False
            else:
                agent_is_observable = [
                    self.agents[agent_id].field_of_view[grid_cell[0], grid_cell[1]]
                    for grid_cell in agent_grid_cells
                ]
                if sum(agent_is_observable) > 0 and not self.equivalent_states(
                    self.agent_states[entity_id], state[entity_id]
                ):
                    # return False, ('agent',1-agent_id)
                    res = False
        else:
            item_id = entity_id - self.num_agents
            item_grid_cells = self.get_perimeter_grid_cells(
                "item", item_id, state[self.num_agents + item_id]
            )
            if (
                sum(
                    [
                        grid_cell[0] < 0
                        or grid_cell[0] > self.room_dim[0]
                        or grid_cell[1] < 0
                        or grid_cell[1] > self.room_dim[1]
                        for grid_cell in item_grid_cells
                    ]
                )
                > 0
            ):
                res = False
            else:
                item_is_observable = [
                    self.agents[agent_id].field_of_view[grid_cell[0], grid_cell[1]]
                    for grid_cell in item_grid_cells
                ]
                if sum(item_is_observable) > 0 and not self.equivalent_states(
                    self.item_states[item_id], state[self.num_agents + item_id]
                ):
                    # return False, ('item',item_id)
                    res = False
        return res

    def resample_inconsistent_entity(
        self,
        agent_id,
        entity_id,
        state,
        all_pos,
        pos_probs,
        fixed_shapes,
        max_attempts=10,
        num_angles=8,
    ):
        """Will only try max_attempts for non overlapping condition and another max_attempts for a relaxed condition"""
        new_state = copy.deepcopy(state)

        # include overlap check
        for attempt_id in range(max_attempts):
            pos_id = np.random.choice(list(range(len(all_pos))), p=pos_probs)
            pos = all_pos[pos_id]
            dA = 360 / num_angles
            angles = [a * dA for a in range(0, num_angles)]
            for angle in angles:
                new_state[entity_id]["pos"] = (int(pos[0]), int(pos[1]))
                new_state[entity_id]["angle"] = math.radians(angle)
                new_state[entity_id]["vel"] = [0.0, 0.0]
                new_state[entity_id]["angleVel"] = 0.0
                new_state[entity_id]["attached"] = None
                new_shape = self.create_shape(entity_id, new_state[entity_id])
                matches = self.belief_matches_observation(
                    agent_id, entity_id, new_state
                )
                overlaps = self.overlaps(new_shape, fixed_shapes)
                if matches and not overlaps:
                    if agent_id == 1 and entity_id == 2:
                        cell = self.world_point_to_grid_cell(
                            new_state[entity_id]["pos"][0],
                            new_state[entity_id]["pos"][1],
                        )
                        # print('resample', new_state[entity_id]['pos'], self.agents[agent_id].certainty[cell[0],cell[1]])
                    return copy.deepcopy(new_state[entity_id])

        # remove overlap check
        for attempt_id in range(max_attempts):
            pos_id = np.random.choice(list(range(len(all_pos))), p=pos_probs)
            pos = all_pos[pos_id]
            dA = 360 / num_angles
            angles = [a * dA for a in range(0, num_angles)]
            for angle in angles:
                new_state[entity_id]["pos"] = (int(pos[0]), int(pos[1]))
                new_state[entity_id]["angle"] = math.radians(angle)
                new_state[entity_id]["vel"] = [0.0, 0.0]
                new_state[entity_id]["angleVel"] = 0.0
                new_state[entity_id]["attached"] = None
                new_shape = self.create_shape(entity_id, new_state[entity_id])
                matches = self.belief_matches_observation(
                    agent_id, entity_id, new_state
                )
                # overlaps = self.overlaps(new_shape, fixed_shapes)
                if matches:
                    # if agent_id == 1 and entity_id == 2:
                    #     cell = self.world_point_to_grid_cell(new_state[entity_id]['pos'][0], new_state[entity_id]['pos'][0])
                    #     print('resample', new_state[entity_id]['pos'], self.agents[agent_id].certainty[cell[0],cell[1]])
                    return copy.deepcopy(new_state[entity_id])

        # if agent_id == 1 and entity_id == 2:
        #     cell = self.world_point_to_grid_cell(new_state[entity_id]['pos'][0], new_state[entity_id]['pos'][0])
        #     print('resample', new_state[entity_id]['pos'], self.agents[agent_id].certainty[cell[0],cell[1]])

        # failed, just a random state
        return copy.deepcopy(new_state[entity_id])

    def create_shape(self, entity_id, state, scale=1.0):
        if entity_id < self.num_agents:
            return create_agent_shape(
                state["pos"], state["angle"], SIZE[self.sizes[entity_id]] * scale
            )
        else:
            return create_item_shape(state["pos"], SIZE[self.sizes[entity_id]] * scale)

    def overlaps(self, shape, shape_list):
        for s in shape_list:
            if shape.intersects(s):  # and not shape.touches(shapes):
                return True
        return False

    def belief_update(self, agent_id, action, nb_steps=None, cInit=None, cBase=None):
        """transition func (simulate one step)"""
        curr_particles = self.agents[agent_id].beliefs.particles
        next_particles = []
        # agent_fov = self.agents[agent_id].field_of_view.copy()
        agent_certainty = self.agents[agent_id].certainty.copy()

        observed_entities = [
            int(obs[1]) + self.num_agents if obs[0] == "item" else int(obs[1])
            for obs in self.agents[agent_id].last_observations.keys()
        ]
        observed_states = [
            self.item_states[obs[1]] if obs[0] == "item" else self.agent_states[obs[1]]
            for obs in self.agents[agent_id].last_observations.keys()
        ]
        observed_shapes = [
            self.create_shape(entity_id, gt_state, scale=1.1)
            for entity_id, gt_state in zip(observed_entities, observed_states)
        ]

        room_dim, world_center = self.room_dim, self.room.position
        x, y = np.meshgrid(
            list(
                range(
                    int(world_center[0] - (room_dim[0]) // 2),
                    int(world_center[0] + (room_dim[0]) // 2),
                )
            ),
            list(
                range(
                    int(world_center[1] + (room_dim[1]) // 2),
                    int(world_center[1] - (room_dim[1]) // 2),
                    -1,
                )
            ),
        )
        all_init_pos = list(zip(*(x.flatten(), y.flatten())))

        # for each particle until get to next n_particles that match FOV
        for curr_state_idx, curr_state in enumerate(curr_particles):
            # step 1: transition
            new_particle = self.particle_transition(agent_id, curr_state, action)

            # step 2: set ground-truth state w.r.t. obs
            # last_observations only current (not all history.)
            for entity_id, gt_state in zip(observed_entities, observed_states):
                # TODO touch sensor in last_observations update..
                """TODO: need a better check on the visibility of attachment!!!"""  # +(attached object can block FOV)
                new_particle[entity_id]["pos"] = gt_state["pos"]
                new_particle[entity_id]["vel"] = gt_state["vel"]
                new_particle[entity_id]["angle"] = gt_state["angle"]
                new_particle[entity_id]["angleVel"] = gt_state["angleVel"]
                new_particle[entity_id]["attached"] = gt_state["attached"]

            # step 3: resample inconsistent entity
            fixed_entities = list(observed_entities)
            mask = agent_certainty.copy()
            fixed_shapes = list(observed_shapes)

            tmp_world = world(gravity=(0, 0), doSleep=True)

            for entity_id in range(self.num_entities):
                if entity_id not in observed_entities:
                    matches = self.belief_matches_observation(
                        agent_id, entity_id, new_particle
                    )
                    curr_shape = self.create_shape(entity_id, new_particle[entity_id])
                    overlaps = self.overlaps(curr_shape, observed_shapes)
                    if matches and not overlaps:
                        fixed_entities.append(entity_id)
                        fixed_shapes.append(curr_shape)
                        if entity_id < self.num_agents:
                            R = SIZE[self.sizes[entity_id]]
                            body = tmp_world.CreateDynamicBody(
                                position=new_particle[entity_id]["pos"],
                                angle=new_particle[entity_id]["angle"],
                                fixtures=b2FixtureDef(
                                    shape=b2PolygonShape(
                                        vertices=[
                                            (-1.5 * R, -R),
                                            (-0.5 * R, R),
                                            (0.5 * R, R),
                                            (1.5 * R, -R),
                                        ]
                                    ),
                                    density=DENSITY[self.densities[entity_id]],
                                    friction=FRICTION,
                                ),
                            )
                            body.linearVelocity.x = new_particle[entity_id]["vel"][0]
                            body.linearVelocity.y = new_particle[entity_id]["vel"][1]
                            body.angularVelocity = new_particle[entity_id]["angleVel"]
                            body.userData = [
                                ("agent", entity_id),
                                self.touch_sensor,
                                None,
                            ]
                            entity_grids = self.get_perimeter_grid_cells(
                                "agent", entity_id, body
                            )
                        else:
                            entity_grids = self.get_perimeter_grid_cells(
                                "item",
                                entity_id - self.num_agents,
                                new_particle[entity_id],
                            )
                        for c in entity_grids:
                            mask[c[0], c[1]] = 1

            # masked_pos = np.array(all_init_pos)[(1-mask).flatten().astype(bool)]
            pos_probs = (1 - mask).flatten()
            pos_probs /= np.sum(pos_probs)
            # if len(masked_pos)==0:
            #     print('masked_pos 0')

            for entity_id in range(self.num_entities):
                if entity_id not in fixed_entities:
                    new_particle[entity_id] = self.resample_inconsistent_entity(
                        agent_id,
                        entity_id,
                        new_particle,
                        all_init_pos,
                        pos_probs,
                        fixed_shapes,
                    )
            next_particles.append(new_particle)

        # next_particles = random.sample(next_particles, min(n_particles,len(next_particles)))
        self.agents[agent_id].beliefs.update(next_particles)
        # print(self.agents[agent_id].beliefs.particles)

    def resample_specified_particles(self, agent_id, particle_idx2resample, goal):
        # print('particle_idx2resample',particle_idx2resample)
        replace_entity_id = GOAL[goal][2]
        curr_particles = self.agents[agent_id].beliefs.particles
        next_particles = copy.deepcopy(curr_particles)
        agent_fov = self.agents[agent_id].field_of_view.copy()
        agent_certainty = self.agents[agent_id].certainty.copy()

        # TODO assuming replace_entity_id is item
        all_entities = [
            ("agent", entity_id) for entity_id in range(self.num_agents)
        ] + [("item", 1 - (self.num_agents - replace_entity_id))]
        fixed_entities = [
            int(obs[1]) + self.num_agents if obs[0] == "item" else int(obs[1])
            for obs in all_entities
        ]

        room_dim, world_center = self.room_dim, self.room.position
        x, y = np.meshgrid(
            list(
                range(
                    int(world_center[0] - (room_dim[0]) // 2),
                    int(world_center[0] + (room_dim[0]) // 2),
                )
            ),
            list(
                range(
                    int(world_center[1] + (room_dim[1]) // 2),
                    int(world_center[1] - (room_dim[1]) // 2),
                    -1,
                )
            ),
        )
        all_init_pos = list(zip(*(x.flatten(), y.flatten())))

        for curr_state_idx in particle_idx2resample:
            new_particle = copy.deepcopy(curr_particles[curr_state_idx])
            fixed_states = [
                new_particle[self.num_agents + obs[1]]
                if obs[0] == "item"
                else new_particle[obs[1]]
                for obs in all_entities
            ]
            fixed_shapes = [
                self.create_shape(entity_id, gt_state)
                for entity_id, gt_state in zip(fixed_entities, fixed_states)
            ]

            # step 3: resample entity
            mask = agent_certainty.copy()
            tmp_world = world(gravity=(0, 0), doSleep=True)
            for entity_id in range(self.num_entities):
                if entity_id != replace_entity_id:
                    if entity_id < self.num_agents:
                        R = SIZE[self.sizes[entity_id]]
                        body = tmp_world.CreateDynamicBody(
                            position=new_particle[entity_id]["pos"],
                            angle=new_particle[entity_id]["angle"],
                            fixtures=b2FixtureDef(
                                shape=b2PolygonShape(
                                    vertices=[
                                        (-1.5 * R, -R),
                                        (-0.5 * R, R),
                                        (0.5 * R, R),
                                        (1.5 * R, -R),
                                    ]
                                ),
                                density=DENSITY[self.densities[entity_id]],
                                friction=FRICTION,
                            ),
                        )
                        body.linearVelocity.x = new_particle[entity_id]["vel"][0]
                        body.linearVelocity.y = new_particle[entity_id]["vel"][1]
                        body.angularVelocity = new_particle[entity_id]["angleVel"]
                        body.userData = [("agent", entity_id), self.touch_sensor, None]
                        entity_grids = self.get_perimeter_grid_cells(
                            "agent", entity_id, body
                        )
                    else:
                        entity_grids = self.get_perimeter_grid_cells(
                            "item", entity_id - self.num_agents, new_particle[entity_id]
                        )
                    for c in entity_grids:
                        mask[c[0], c[1]] = 1

            masked_pos = np.array(all_init_pos)[(1 - mask).flatten().astype(bool)]
            new_particle[replace_entity_id] = self.resample_inconsistent_entity(
                agent_id, replace_entity_id, new_particle, masked_pos, fixed_shapes
            )
            next_particles[curr_state_idx] = new_particle

        self.agents[agent_id].beliefs.update(next_particles)

    def particle_transition(
        self, agent_id, curr_state, action, nb_steps=None, cInit=None, cBase=None
    ):
        """transition func (simulate one step)"""
        self._setup_tmp(self.env_id, curr_state)
        self._send_action_tmp(agent_id, action)
        # if not self.in_field_of_view(agent_id,('agent',1-agent_id)):
        #     random_action = random.choice(self.action_space)
        #     # print('random_action',random_action)
        #     self._send_action_tmp(1-agent_id, random_action) #TODO
        self._step_tmp(update_certainty=False)
        next_state = self._get_state_tmp()
        return next_state

    def transition(
        self,
        agent_id,
        curr_state,
        curr_certainty,
        action,
        expected_action,
        cInit=None,
        cBase=None,
        update_certainty=True,
    ):
        """transition func (simulate one step)"""
        # TODO: tmp setup or real? does it matter?
        # start = time.time()
        self._setup_tmp(self.env_id, curr_state, curr_certainty)
        self._send_action_tmp(agent_id, action)
        if self.num_agents == 2:
            other_agent_id = 1 - agent_id
            self._send_action_tmp(other_agent_id, expected_action)
        """TODO: do not exclude these actions"""
        self._step_tmp(
            update_certainty=False
        )  # update_certainty)# and action not in [None, 'stop', 'noforce', 'attach', 'detach'])
        # print(curr_agents_pos, action, self.tmp_agents_pos)
        next_state, next_certainty = self._get_state_tmp(agent_id=agent_id)
        # print(curr_state, action, expected_action, next_state)

        # end = time.time()
        # print(end - start)

        return next_state, next_certainty

    def set_state(self, agent_states, item_states, certainty, FOVs, steps):
        """set the state of the environment"""
        for agent_id in range(self.num_agents):
            if self.attached[agent_id] is not None:
                # print(self.attached[agent_id])
                self.world.DestroyJoint(self.attached[agent_id])
                self.attached[agent_id] = None
                # self.items[0].userData[-1] = None
                self.items[self.agents[agent_id].userData[-1]].userData[
                    -1
                ] = None  # TODO item id
                self.agents[agent_id].userData[-1] = None

        for agent_id, agent_state in enumerate(agent_states):
            self.world.DestroyBody(self.agents[agent_id])
            R = SIZE[self.sizes[agent_id]]
            self.agents[agent_id] = self.world.CreateDynamicBody(
                position=(agent_state["pos"][0], agent_state["pos"][1]),
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(
                        vertices=[
                            (-1.5 * R, -R),
                            (-0.5 * R, R),
                            (0.5 * R, R),
                            (1.5 * R, -R),
                        ]
                    ),
                    density=DENSITY[self.densities[agent_id]],
                    friction=FRICTION,
                ),
                angle=agent_state["angle"],
            )
            self.agents[agent_id].linearVelocity.x = agent_state["vel"][0]
            self.agents[agent_id].linearVelocity.y = agent_state["vel"][1]
            self.agents[agent_id].angularVelocity = agent_state["angleVel"]
            self.agents[agent_id].userData = [
                ("agent", agent_id),
                self.touch_sensor,
                None,
            ]
            self.agents[agent_id].certainty = certainty[agent_id].copy()
            self.agents[agent_id].field_of_view = FOVs[agent_id].copy()
        # self.update_field_of_view()

        for item_id, item_state in enumerate(item_states):
            self.world.DestroyBody(self.items[item_id])
            self.items[item_id] = self.world.CreateDynamicBody(
                position=(item_state["pos"][0], item_state["pos"][1]),
                angle=item_state["angle"],
            )
            self.items[item_id].CreateCircleFixture(
                radius=SIZE[self.sizes[self.num_agents + item_id]],
                density=DENSITY[self.densities[self.num_agents + item_id]],
                friction=FRICTION,
                restitution=0,
            )
            self.items[item_id].linearVelocity.x = item_state["vel"][0]
            self.items[item_id].linearVelocity.y = item_state["vel"][1]
            self.items[item_id].angularVelocity = item_state["angleVel"]
            self.items[item_id].userData = [("item", item_id), None]

        for agent_id in range(self.num_agents):
            # if self.attached[agent_id] is not None:
            #     self.world.DestroyJoint(self.attached[agent_id])
            #     self.attached[agent_id] = None
            if agent_states[agent_id]["attached"] is not None:
                # if attached[agent_id] is not None:
                # if agent_states[agent_id]['attached'] is None:
                #     print('set_state attached')
                f = {"spring": 0.3, "rope": 0.1, "rod": 100}
                d = {"spring": 0, "rope": 0, "rod": 0.5}
                dfn = b2DistanceJointDef(
                    frequencyHz=f["rod"],
                    dampingRatio=d["rod"],
                    bodyA=self.agents[agent_id],
                    bodyB=self.items[
                        agent_states[agent_id]["attached"]
                    ],  # TODO: attached item id!!! #self.items[0],
                    localAnchorA=(0, 0),
                    localAnchorB=(0, 0),
                )
                self.attached[agent_id] = self.world.CreateJoint(dfn)
                self.agents[agent_id].userData[-1] = agent_states[agent_id][
                    "attached"
                ]  # TODO: attached item id!!!
                self.items[item_id].userData[-1] = agent_id

        self.agent_states = [_get_state_from_body(body) for body in self.agents]
        self.item_states = [_get_state_from_body(body) for body in self.items]
        self.steps = steps

    def belief_render_all_particles_separately(
        self, render_agent_id, curr_particles, record_dir="", path_extension=""
    ):
        """render the tmp environment, based on render"""
        colors = self.colors
        mask = self.agents[render_agent_id].field_of_view * 255
        mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
        mask = np.reshape(mask, mask.shape + (1,))
        mask = np.repeat(mask, 3, axis=2)
        for p_id, particle in enumerate(curr_particles):
            # print('particle',particle)
            self._setup_tmp(self.env_id, particle)
            self.tmp_screen.fill((255, 255, 255, 255))  # TODO same screen?
            for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
                _my_draw_patch(
                    lm_loc,
                    self.tmp_screen,
                    color,
                    self.PPM,
                    self.SCREEN_WIDTH,
                    self.SCREEN_HEIGHT,
                )
            for body_id, body in enumerate(
                [self.tmp_room] + self.tmp_agents + self.tmp_items
            ):
                if body_id and not self.visibility[body_id - 1]:
                    continue
                for fixture in body.fixtures:
                    fixture.shape.draw(
                        self.tmp_screen,
                        body,
                        fixture,
                        colors[body_id],
                        self.PPM,
                        self.SCREEN_HEIGHT,
                    )
            pygame.display.flip()
            cropped = pygame.display.get_surface().subsurface((120, 40, 400, 400))
            frame = pygame.surfarray.array3d(cropped)
            frame = np.transpose(frame, (1, 0, 2))
            # combined = np.uint8(0.7*(frame)+0.3*(mask))
            combined = np.uint8(1.0 * (frame))

            results_dir = (
                record_dir
                + "/belief_snapshots/"
                + str(render_agent_id)
                + "bel"
                + path_extension
                + "/"
                + str(self.steps)
                + "agent"
                + str(render_agent_id)
                + "belief/"
            )
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.imsave(results_dir + str(p_id) + ".png", combined)

    def belief_render_all_particles(
        self, render_agent_id, curr_particles, record_dir="", path_extension=""
    ):
        """render the tmp environment, based on render"""
        colors = self.colors
        all_particles_frame = np.zeros((400, 400, 3))
        for particle in curr_particles:
            # print('particle',particle)
            self._setup_tmp(self.env_id, particle)
            self.tmp_screen.fill((255, 255, 255, 255))  # TODO same screen?
            for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
                _my_draw_patch(
                    lm_loc,
                    self.tmp_screen,
                    color,
                    self.PPM,
                    self.SCREEN_WIDTH,
                    self.SCREEN_HEIGHT,
                )
            for body_id, body in enumerate(
                [self.tmp_room] + self.tmp_agents + self.tmp_items
            ):
                if body_id and not self.visibility[body_id - 1]:
                    continue
                for fixture in body.fixtures:
                    fixture.shape.draw(
                        self.tmp_screen,
                        body,
                        fixture,
                        colors[body_id],
                        self.PPM,
                        self.SCREEN_HEIGHT,
                    )
            pygame.display.flip()

            cropped = pygame.display.get_surface().subsurface((120, 40, 400, 400))
            frame = pygame.surfarray.array3d(cropped)
            frame = np.transpose(frame, (1, 0, 2))
            all_particles_frame += frame

        all_particles_frame = all_particles_frame / np.max(all_particles_frame) * 255
        mask = self.agents[render_agent_id].field_of_view * 255
        mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
        mask = np.reshape(mask, mask.shape + (1,))
        mask = np.repeat(mask, 3, axis=2)
        combined = np.uint8(0.7 * (all_particles_frame) + 0.3 * (mask))

        results_dir = (
            record_dir
            + "/belief_snapshots/"
            + str(render_agent_id)
            + "bel"
            + path_extension
            + "/"
        )
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.imsave(
            results_dir
            + str(self.steps)
            + "agent"
            + str(render_agent_id)
            + "belief.png",
            combined,
        )

    def belief_render(self, render_agent_id, record_dir="", path_extension=""):
        """render the tmp environment, based on render"""
        colors = self.colors

        # print('belief render',path_extension)
        # [print(f.shape.all_vertices) for f in self.tmp_room.fixtures]

        self.tmp_screen.fill((255, 255, 255, 255))  # TODO same screen?
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(
                lm_loc,
                self.tmp_screen,
                color,
                self.PPM,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
            )
        for body_id, body in enumerate(
            [self.tmp_room] + self.tmp_agents + self.tmp_items
        ):
            if body_id and not self.visibility[body_id - 1]:
                continue
            for fixture in body.fixtures:
                fixture.shape.draw(
                    self.tmp_screen,
                    body,
                    fixture,
                    colors[body_id],
                    self.PPM,
                    self.SCREEN_HEIGHT,
                )
        pygame.display.flip()

        cropped = pygame.display.get_surface().subsurface((120, 40, 400, 400))
        frame = pygame.surfarray.array3d(cropped)
        frame = np.transpose(frame, (1, 0, 2))
        mask = self.agents[render_agent_id].field_of_view * 255
        mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
        mask = np.reshape(mask, mask.shape + (1,))
        mask = np.repeat(mask, 3, axis=2)
        combined = np.uint8(0.5 * (frame) + 0.5 * (mask))
        # plt.imshow(combined)
        # plt.show()
        results_dir = (
            record_dir
            + "/belief_snapshots/"
            + str(render_agent_id)
            + "bel"
            + path_extension
            + "/"
        )
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.imsave(
            results_dir
            + str(self.steps)
            + "agent"
            + str(render_agent_id)
            + "belief.png",
            combined,
        )

    def render_FOV(self, record_dir="", path_extension=""):
        """render the environment"""
        colors = self.colors

        self.screen.fill((255, 255, 255, 255))
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(
                lm_loc,
                self.screen,
                color,
                self.PPM,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
            )
        for body_id, body in enumerate([self.room] + self.agents + self.items):
            if body_id and not self.visibility[body_id - 1]:
                continue
            for fixture in body.fixtures:
                fixture.shape.draw(
                    self.screen,
                    body,
                    fixture,
                    colors[body_id],
                    self.PPM,
                    self.SCREEN_HEIGHT,
                )
        pygame.display.flip()

        # cropped = pygame.display.get_surface().subsurface((117.5,37.5,405,405))
        cropped = pygame.display.get_surface().subsurface((120, 40, 400, 400))
        frame = pygame.surfarray.array3d(cropped)
        frame = np.transpose(frame, (1, 0, 2))
        for agent_id, agent in enumerate(self.agents):
            mask = agent.field_of_view * 255
            mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
            mask = np.reshape(mask, mask.shape + (1,))
            mask = np.repeat(mask, 3, axis=2)
            combined = np.uint8(0.5 * (frame) + 0.5 * (mask))
            # plt.imshow(combined)
            # plt.show()
            results_dir = (
                record_dir
                + "/belief_snapshots/"
                + str(agent_id)
                + "FOV"
                + path_extension
                + "/"
            )
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.imsave(
                results_dir + str(self.steps) + "agent" + str(agent_id) + "FOV.png",
                combined,
            )

    def render(self):
        """render the environment"""
        colors = self.colors

        self.screen.fill((255, 255, 255, 255))
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(
                lm_loc,
                self.screen,
                color,
                self.PPM,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
            )
        for body_id, body in enumerate([self.room] + self.agents + self.items):
            if body_id and not self.visibility[body_id - 1]:
                continue
            for fixture in body.fixtures:
                fixture.shape.draw(
                    self.screen,
                    body,
                    fixture,
                    colors[body_id],
                    self.PPM,
                    self.SCREEN_HEIGHT,
                )
        pygame.display.flip()

        self.clock.tick(self.TARGET_FPS)
        if self.video:
            obs = _get_obs(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            curr_frame = cv2.cvtColor(obs.transpose(1, 0, 2), cv2.COLOR_RGB2BGR)
            self.video.write(curr_frame)

    def render_replay(self):
        """render the environment"""
        self._setup_tmp(
            self.env_id, self.get_state()
        )  # will flip angles & pos & velocity
        colors = self.colors
        self.tmp_screen.fill((255, 255, 255, 255))
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(
                lm_loc,
                self.tmp_screen,
                color,
                self.PPM,
                self.SCREEN_WIDTH,
                self.SCREEN_HEIGHT,
            )
        for body_id, body in enumerate(
            [self.tmp_room] + self.tmp_agents + self.tmp_items
        ):
            if body_id and not self.visibility[body_id - 1]:
                continue
            for fixture in body.fixtures:
                fixture.shape.draw(
                    self.tmp_screen,
                    body,
                    fixture,
                    colors[body_id],
                    self.PPM,
                    self.SCREEN_HEIGHT,
                )
        pygame.display.flip()

        self.clock.tick(self.TARGET_FPS)
        if self.video:
            obs = _get_obs(self.tmp_screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            curr_frame = cv2.cvtColor(obs.transpose(1, 0, 2), cv2.COLOR_RGB2BGR)
            self.video.write(curr_frame)

    def replay(self, trajectories, record_path=None, order=0):
        """replay an old espisode based on recorded trajectories"""
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
            )
        self.world = world(gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
                (-self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, self.room_dim[1] / 2),
                (self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
                (-self.room_dim[0] / 2, self.room_dim[1] / 2 - self.door_length),
            ]
        )
        self.agents, self.items = [], []
        self.densities = [1, 1]
        T = len(trajectories[0])
        for t in range(T):
            for body in self.agents:
                self.world.DestroyBody(body)
            for body in self.items:
                self.world.DestroyBody(body)
            self.agents, self.items = [], []
            agent_indices = [0, 1] if order == 0 else [1, 0]
            for agent_id in agent_indices:
                body = self.world.CreateDynamicBody(
                    position=(
                        trajectories[agent_id][t][0],
                        trajectories[agent_id][t][1],
                    )
                )
                body.CreateCircleFixture(
                    radius=1, density=self.densities[0], friction=FRICTION
                )
                self.agents.append(body)
            body = self.world.CreateDynamicBody(
                position=(trajectories[2][t][0], trajectories[2][t][1])
            )
            body.CreateCircleFixture(
                radius=2, density=self.densities[0], friction=FRICTION
            )
            self.items.append(body)
            self.render()

    def release(self):
        """release video writer"""
        if self.video:
            # self.step()
            self.video.release()
            self.video = None

    def destroy(self):
        """destroy the environment"""
        self.world.DestroyBody(self.room)
        self.room = None
        for agent in self.agents:
            self.world.DestroyBody(agent)
        self.agents = []
        if self.enable_renderer:
            pygame.quit()

    # shape_def is agent vertices or radius & item center
    def get_perimeter_grid_cells(self, entity_type, entity_id, state):
        # state only of entity type, id accordingly
        if entity_type == "agent":
            raw_grid_cells = []
            shape_def = [
                state.GetWorldPoint(localPoint=v)
                for v in state.fixtures[0].shape.vertices
            ]
            n_Vs = len(shape_def)
            int_vs = [(int(np.floor(v[0])), int(np.floor(v[1]))) for v in shape_def]
            for i, v in enumerate(int_vs):
                raw_grid_cells += list(
                    bresenham(
                        v[0], v[1], int_vs[(i + 1) % n_Vs][0], int_vs[(i + 1) % n_Vs][1]
                    )
                )
            # print(shape_def)
            # print(grid_cells)
            # print('bres')
        elif entity_type == "item":
            R, origin = SIZE[self.sizes[self.num_agents + entity_id]], state["pos"]
            N_samples = 16
            alpha = 2 * np.pi / N_samples
            raw_grid_cells = [
                (R * np.cos(k * alpha) + origin[0], R * np.sin(k * alpha) + origin[1])
                for k in range(N_samples)
            ]

        grid_cells = [self.world_point_to_grid_cell(g[0], g[1]) for g in raw_grid_cells]
        grid_cells = np.unique(grid_cells, axis=0)

        # if np.sum(np.unique(grid_cells)>=20) > 0:
        #     print('grid cells perimeter input', shape_def)
        #     print('grid cells perimeter output raw', raw_grid_cells)
        #     print('grid cells perimeter output',grid_cells)
        return grid_cells

    # vector direction from start point along gaze ray, size 1
    def line_to_grid_cells(self, start_point, vector):
        max_length = (self.room_dim[0] ** 2 + self.room_dim[1] ** 2) ** 0.5
        # from start_point going in vector dir (should be of size 1 with orientation line) get new point
        grid_cells = []
        cur_point = start_point
        while (
            math.hypot(cur_point[0] - start_point[0], cur_point[1] - start_point[1])
            <= max_length
        ):
            cur_cell = self.world_point_to_grid_cell(cur_point[0], cur_point[1])
            if [cur_cell[0], cur_cell[1]] not in grid_cells and (
                cur_cell[0] >= 0
                and cur_cell[0] < self.room_dim[0]
                and cur_cell[1] >= 0
                and cur_cell[1] < self.room_dim[1]
            ):
                grid_cells.append([cur_cell[0], cur_cell[1]])
                cur_point = cur_point + vector
        return np.array(grid_cells)

    def world_pos_in_cone(self, c, agent_id):
        agent = self.agents[agent_id]
        p1 = agent.worldCenter
        Vs = agent.fixtures[0].shape.vertices
        eye_pos = [
            (
                (Vs[0][0] - Vs[1][0]) * 0.3 + Vs[1][0],
                (Vs[0][1] - Vs[1][1]) * 0.3 + Vs[1][1],
            ),
            (
                (Vs[3][0] - Vs[2][0]) * 0.3 + Vs[2][0],
                (Vs[3][1] - Vs[2][1]) * 0.3 + Vs[2][1],
            ),
        ]
        p2 = agent.GetWorldPoint(localPoint=(eye_pos[0][0], eye_pos[0][1]))  # right eye
        p3 = agent.GetWorldPoint(localPoint=(0.0, eye_pos[0][1]))  # middle
        cone_boundary_vec = b2Vec2(p2[0] - p1[0], p2[1] - p1[1])
        cone_mid_vec = b2Vec2(p3[0] - p1[0], p3[1] - p1[1])
        cone_angle = _get_angle(cone_boundary_vec, cone_mid_vec)  # half angle
        return _get_angle(cone_mid_vec, (c[0] - p1[0], c[1] - p1[1])) < cone_angle

    def set_field_of_view(self, agent):
        p1 = agent.worldCenter  # body center
        Vs = agent.fixtures[0].shape.vertices
        eye_pos = [
            (
                (Vs[0][0] - Vs[1][0]) * 0.3 + Vs[1][0],
                (Vs[0][1] - Vs[1][1]) * 0.3 + Vs[1][1],
            ),
            (
                (Vs[3][0] - Vs[2][0]) * 0.3 + Vs[2][0],
                (Vs[3][1] - Vs[2][1]) * 0.3 + Vs[2][1],
            ),
        ]
        p2 = agent.GetWorldPoint(localPoint=(eye_pos[0][0], eye_pos[0][1]))  # right eye
        p3 = agent.GetWorldPoint(localPoint=(0.0, eye_pos[0][1]))  # middle
        cone_boundary_vec = b2Vec2(p2[0] - p1[0], p2[1] - p1[1])
        cone_mid_vec = b2Vec2(p3[0] - p1[0], p3[1] - p1[1])

        # print(math.degrees(math.atan2(cone_mid_vec[1], cone_mid_vec[0])), math.degrees(agent.angle), math.degrees(math.atan2(math.sin(agent.angle), math.cos(agent.angle)) + math.pi * 0.5))

        agent.field_of_view = np.zeros(self.room_dim)
        # move to center of cell world point
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5
        grid_cells_world_center = [
            [col_baseline + c, row_baseline - r]
            for r in range(self.room_dim[0])
            for c in range(self.room_dim[1])
        ]  # x,y
        cone_angle = _get_angle(cone_boundary_vec, cone_mid_vec)  # half angle
        in_cone_idx = [
            _get_angle(cone_mid_vec, (c[0] - p1[0], c[1] - p1[1])) < cone_angle
            for c in grid_cells_world_center
        ]
        in_cone_y, in_cone_x = np.meshgrid(
            np.arange(self.room_dim[0]), np.arange(self.room_dim[1])
        )
        agent.field_of_view[
            in_cone_x.flatten()[in_cone_idx], in_cone_y.flatten()[in_cone_idx]
        ] = 1

    # is a cell visible
    def is_visible(self, cell_center, agent_id):
        collisions = self.calculate_collision(
            self.agents[agent_id].worldCenter, cell_center, agent_id
        )
        # if len(collisions) != 0:
        #     print('cell',cell_center,'col',collisions)
        return len(collisions) == 0

    def is_center_visible(self, cell_center, agent_id):
        collisions = self.calculate_collision_walls(
            self.agents[agent_id].worldCenter, cell_center, agent_id
        )
        return len(collisions) == 0

    def _is_visible_tmp(self, cell_center, agent_id):
        collisions = self._calculate_collision_tmp(
            self.tmp_agents[agent_id].worldCenter, cell_center, agent_id
        )
        return len(collisions) == 0

    def _update_FOV(self, agent_id):
        cone_idx = np.where(self.agents[agent_id].field_of_view)
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5
        cone_world_center = [
            [col_baseline + c, row_baseline - r]
            for r, c in zip(cone_idx[0], cone_idx[1])
        ]  # x,y
        visible_in_cone = np.array(
            [
                self.is_visible(cell_center, agent_id)
                for cell_center in cone_world_center
            ]
        ).astype(int)
        self.agents[agent_id].field_of_view[cone_idx] = visible_in_cone

    def _update_FOV_tmp(self, agent_id):
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5

        p1 = self.tmp_agents[agent_id].worldCenter
        Vs = self.tmp_agents[agent_id].fixtures[0].shape.vertices
        eye_pos = [
            (
                (Vs[0][0] - Vs[1][0]) * 0.3 + Vs[1][0],
                (Vs[0][1] - Vs[1][1]) * 0.3 + Vs[1][1],
            ),
            (
                (Vs[3][0] - Vs[2][0]) * 0.3 + Vs[2][0],
                (Vs[3][1] - Vs[2][1]) * 0.3 + Vs[2][1],
            ),
        ]
        p3 = self.tmp_agents[agent_id].GetWorldPoint(
            localPoint=(0.0, eye_pos[0][1])
        )  # middle
        cone_mid_vec = b2Vec2(p3[0] - p1[0], p3[1] - p1[1])
        angle = math.atan2(cone_mid_vec[1], cone_mid_vec[0]) + math.pi

        num_angles = len(self.viz_map)
        dA = 2 * math.pi / num_angles
        a = int(angle // dA)
        r1 = int(row_baseline - self.tmp_agents[agent_id].worldCenter[1])
        c1 = int(self.tmp_agents[agent_id].worldCenter[0] - col_baseline)
        p1 = self.tmp_agents[agent_id].worldCenter

        self.tmp_agents[agent_id].field_of_view = np.array(self.viz_map[a, r1, c1, :])

    def _create_visibility_map(self):
        row_baseline = 12 + (self.room_dim[0] // 2) - 0.5
        col_baseline = 16 - (self.room_dim[1] // 2) + 0.5

        room_center = [16, 12]
        doors_pos = self.doors_pos_start
        maze_walls = [
            (room_center, doors_pos[0]),
            (doors_pos[1], room_center),
            (doors_pos[2], room_center),
            (room_center, doors_pos[3]),
        ]
        min_x, min_y, max_x, max_y = (
            -self.room_dim[0] / 2 + 16,
            self.room_dim[0] / 2 + 16,
            -self.room_dim[1] / 2 + 12,
            self.room_dim[1] / 2 + 12,
        )
        room_walls = [
            ((min_x, min_y), (min_x, max_y)),
            ((max_x, min_y), (max_x, max_y)),
            ((min_x, min_y), (max_x, min_y)),
            ((min_x, max_y), (max_x, max_y)),
        ]

        cone_angle = 0.9544987278903081

        num_angles = 16
        dA = 2 * math.pi / num_angles

        self.viz_map = np.zeros((num_angles,) + self.room_dim + self.room_dim)

        for a in range(num_angles):
            agent_angle = a * dA - math.pi
            cone_mid_vec = b2Vec2(math.cos(agent_angle), math.sin(agent_angle))
            # print(agent_angle, cone_mid_vec)
            for r1 in range(self.room_dim[0]):
                for c1 in range(self.room_dim[1]):
                    for r2 in range(self.room_dim[0]):
                        for c2 in range(self.room_dim[1]):
                            if r1 == r2 and c1 == c2:
                                self.viz_map[a][r1][c1][r2][c2] = 1
                                continue

                            p1 = [col_baseline + c1, row_baseline - r1]
                            p2 = [col_baseline + c2, row_baseline - r2]

                            if (
                                _get_angle(cone_mid_vec, (p2[0] - p1[0], p2[1] - p1[1]))
                                >= cone_angle
                            ):
                                continue

                            if self._get_room_id(p1) == self._get_room_id(p2):
                                self.viz_map[a][r1][c1][r2][c2] = 1
                                # self.viz_map[a][r2][c2][r1][c1] = 1
                                continue

                            no_collision = True
                            for wall in maze_walls + room_walls:
                                (x, y, valid, r, s) = _get_segment_intersection(
                                    p1, p2, wall[0], wall[1]
                                )
                                if (
                                    valid
                                    and r >= 0
                                    and _not_door(round(x, 2), round(y, 2), doors_pos)
                                    and _point_in_room(
                                        (round(x, 2), round(y, 2)), self.room
                                    )
                                    and _get_dist(p1, p2) >= _get_dist((x, y), p2)
                                    and _get_dist(p1, p2) >= _get_dist((x, y), p1)
                                ):
                                    no_collision = False
                                    break
                            if no_collision:
                                self.viz_map[a][r1][c1][r2][c2] = 1
                                # self.viz_map[a][r2][c2][r1][c1] = 1

    def calculate_collision_walls(self, p1, p2, agent_id):
        collision_points = []
        # maze walls
        room_center = self.room.position
        doors_pos = self.doors_pos_start
        maze_walls = [
            (room_center, doors_pos[0]),
            (doors_pos[1], room_center),
            (doors_pos[2], room_center),
            (room_center, doors_pos[3]),
        ]
        # room walls
        min_x, min_y, max_x, max_y = self.room_bound
        room_walls = [
            ((min_x, min_y), (min_x, max_y)),
            ((max_x, min_y), (max_x, max_y)),
            ((min_x, min_y), (max_x, min_y)),
            ((min_x, max_y), (max_x, max_y)),
        ]
        wall_intersections = [
            _get_segment_intersection(p1, p2, wall[0], wall[1])
            for wall in maze_walls + room_walls
        ]
        [
            collision_points.append((x, y))
            for (x, y, valid, r, s) in wall_intersections
            if valid
            and r >= 0
            and _not_door(round(x, 2), round(y, 2), doors_pos)
            and _point_in_room((round(x, 2), round(y, 2)), self.room)
            and _get_dist(p1, p2) >= _get_dist((x, y), p2)
            and _get_dist(p1, p2) >= _get_dist((x, y), p1)
        ]
        # print(collision_points)
        if len(collision_points) != 0:
            return [
                collision_points[
                    np.argmin([_get_dist(p1, col) for col in collision_points])
                ]
            ]
        else:
            return collision_points

    def calculate_collision(self, p1, p2, agent_id):
        collision_points = []
        # other agent (4 lines) and items (circle) --> can ask if center dist <= R and set center as collision
        for other_agent_id in range(self.num_agents):
            if other_agent_id == agent_id:
                continue
            other_agent_center = self.agents[other_agent_id].worldCenter
            dist = _get_point_dist_from_seg(
                [p1[0], p1[1]],
                [p2[0], p2[1]],
                [other_agent_center[0], other_agent_center[1]],
            )
            R = SIZE[self.sizes[other_agent_id]]
            is_on_ray = _get_dist(p1, p2) >= _get_dist(
                other_agent_center, p2
            ) and _get_dist(p1, p2) >= _get_dist(other_agent_center, p1)
            if dist <= R and is_on_ray:
                collision_points.append(other_agent_center)
        # items
        for item_id in range(self.num_items):
            item_center = self.items[item_id].worldCenter
            dist = _get_point_dist_from_seg(
                [p1[0], p1[1]], [p2[0], p2[1]], [item_center[0], item_center[1]]
            )
            R = SIZE[self.sizes[self.num_agents + item_id]]
            is_on_ray = _get_dist(p1, p2) >= _get_dist(item_center, p2) and _get_dist(
                p1, p2
            ) >= _get_dist(item_center, p1)
            if dist <= R and is_on_ray:
                collision_points.append(item_center)
        # maze walls
        room_center = self.room.position
        # doors_pos = [(16, 1.4 * (self.room_dim[1] / 4) + 12), (-self.room_dim[0] / 4 + 16, 12),
        #              (16, -self.room_dim[1] / 4 + 12), (self.room_dim[0] / 4 + 16, 12)] #old env-id 16
        # doors_pos[0] = (16,22) #TODO not general. env_id 16
        doors_pos = self.doors_pos_start
        maze_walls = [
            (room_center, doors_pos[0]),
            (doors_pos[1], room_center),
            (doors_pos[2], room_center),
            (room_center, doors_pos[3]),
        ]
        # room walls
        # min_x, min_y, max_x, max_y = _get_room_bound(self.room)
        min_x, min_y, max_x, max_y = self.room_bound
        room_walls = [
            ((min_x, min_y), (min_x, max_y)),
            ((max_x, min_y), (max_x, max_y)),
            ((min_x, min_y), (max_x, min_y)),
            ((min_x, max_y), (max_x, max_y)),
        ]
        wall_intersections = [
            _get_segment_intersection(p1, p2, wall[0], wall[1])
            for wall in maze_walls + room_walls
        ]
        [
            collision_points.append((x, y))
            for (x, y, valid, r, s) in wall_intersections
            if valid
            and r >= 0
            and _not_door(round(x, 2), round(y, 2), doors_pos)
            and _point_in_room((round(x, 2), round(y, 2)), self.room)
            and _get_dist(p1, p2) >= _get_dist((x, y), p2)
            and _get_dist(p1, p2) >= _get_dist((x, y), p1)
        ]
        # print(collision_points)
        if len(collision_points) != 0:
            return [
                collision_points[
                    np.argmin([_get_dist(p1, col) for col in collision_points])
                ]
            ]
        else:
            return collision_points

    def _calculate_collision_tmp(self, p1, p2, agent_id):
        collision_points = []
        # other agent (4 lines) and items (circle) --> can ask if center dist <= R and set center as collision
        for other_agent_id in range(self.num_agents):
            if other_agent_id == agent_id:
                continue
            other_agent_center = self.tmp_agents[other_agent_id].worldCenter
            dist = _get_point_dist_from_seg(
                [p1[0], p1[1]],
                [p2[0], p2[1]],
                [other_agent_center[0], other_agent_center[1]],
            )
            R = SIZE[self.sizes[other_agent_id]]
            is_on_ray = _get_dist(p1, p2) >= _get_dist(
                other_agent_center, p2
            ) and _get_dist(p1, p2) >= _get_dist(other_agent_center, p1)
            if dist <= R and is_on_ray:
                collision_points.append(other_agent_center)
                return collision_points
        # items
        for item_id in range(self.num_items):
            item_center = self.tmp_items[item_id].worldCenter
            dist = _get_point_dist_from_seg(
                [p1[0], p1[1]], [p2[0], p2[1]], [item_center[0], item_center[1]]
            )
            R = SIZE[self.sizes[self.num_agents + item_id]]
            is_on_ray = _get_dist(p1, p2) >= _get_dist(item_center, p2) and _get_dist(
                p1, p2
            ) >= _get_dist(item_center, p1)
            if dist <= R and is_on_ray:
                collision_points.append(item_center)
                return collision_points
        # #maze walls
        # room_center = self.room.position
        # # doors_pos = [(16, 1.4 * (self.room_dim[1] / 4) + 12), (-self.room_dim[0] / 4 + 16, 12),
        # #              (16, -self.room_dim[1] / 4 + 12), (self.room_dim[0] / 4 + 16, 12)] #old env-id 16
        # # doors_pos[0] = (16,22) #TODO not general. env_id 16
        # doors_pos = self.doors_pos_start
        # maze_walls = [(room_center, doors_pos[0]), (doors_pos[1], room_center),
        #               (doors_pos[2], room_center), (room_center, doors_pos[3])]
        # #room walls
        # min_x, min_y, max_x, max_y = -self.room_dim[0]/2 + 16, self.room_dim[0]/2 + 16, -self.room_dim[1]/2 + 12, self.room_dim[1]/2 + 12 #_get_room_bound(self.room)
        # room_walls = [((min_x, min_y),(min_x, max_y)), ((max_x, min_y),(max_x, max_y)),
        #               ((min_x, min_y),(max_x, min_y)), ((min_x, max_y),(max_x, max_y))]
        # wall_intersections = [_get_segment_intersection(p1, p2, wall[0], wall[1]) for wall in maze_walls+room_walls]
        # for (x, y, valid, r, s) in wall_intersections:
        #     if valid and r >= 0 and \
        #      _not_door(round(x,2),round(y,2),doors_pos) and _point_in_room((round(x,2),round(y,2)),self.room) and \
        #      _get_dist(p1, p2) >= _get_dist((x,y), p2) and _get_dist(p1, p2) >= _get_dist((x,y), p1):
        #      return (x,y)

        return []

    def update_field_of_view(self):
        for agent_id, agent in enumerate(self.agents):
            if self.full_obs[agent_id]:
                agent.field_of_view = np.ones(self.room_dim)
            else:
                self.set_field_of_view(agent)
                self._update_FOV(agent_id)
            # update confidence
            if hasattr(agent, "certainty"):
                agent.certainty *= 1.0 - self.temporal_decay[agent_id]
                agent.certainty += agent.field_of_view
                agent.certainty = agent.certainty.clip(0, 1)
                # agent.certainty /= np.sum(agent.certainty)
                # plt.imshow(agent.certainty)
                # plt.show()

    def _update_field_of_view_tmp(self):
        """for imaginary agents"""
        for agent_id, agent in enumerate(self.tmp_agents):
            if self.full_obs[agent_id]:
                agent.field_of_view = np.ones_like(agent.field_of_view)
            else:
                # start = time.time()
                # self.set_field_of_view(agent)
                # end = time.time()
                # print('set field of view:', end - start)
                # start = time.time()
                self._update_FOV_tmp(agent_id)
                # end = time.time()
                # print('update FOV:', end - start)
            # update confidence
            if hasattr(agent, "certainty"):
                agent.certainty *= 1.0 - self.temporal_decay[agent_id]
                agent.certainty += agent.field_of_view
                # agent.certainty /= np.sum(agent.certainty)
                agent.certainty = agent.certainty.clip(0, 1)

    def world_point_to_grid_cell(self, world_x, world_y):
        orig = world_x, world_y
        world_x, world_y = math.floor(world_x * 10) / 10, math.floor(world_y * 10) / 10
        # world_x, world_y = round(world_x,1), round(world_y,1)
        col_baseline = self.room.position.x - (self.room_dim[0] // 2)
        row_baseline = self.room.position.y - (self.room_dim[1] // 2)

        grid_col = int(np.floor(world_x - col_baseline))
        grid_row = int(self.room_dim[0]) - 1 - int(np.floor(world_y - row_baseline))

        if grid_col == self.room_dim[0]:
            grid_col -= 1
        if grid_row == self.room_dim[1]:
            grid_row -= 1
        if grid_col == -1:
            grid_col += 1
        if grid_row == -1:
            grid_row += 1

        # if grid_col<0 or grid_row<0:
        #     print('world_point_to_grid_cell -1')
        # raise TypeError('world_point_to_grid_cell -1')

        # # grid_row, grid_col = math.ceil(row_baseline - world_y), math.ceil(world_x - col_baseline)
        # grid_row, grid_col = int(round(row_baseline - world_y,0)), int(round(world_x - col_baseline,0))
        # if abs(world_y-row_baseline) != 0.0: #> 0.1:
        #     grid_row -= 1
        # if abs(world_x-col_baseline) != 0.0: #> 0.1:
        #     grid_col -= 1
        # # print(world_x, world_y, '-->',grid_row, grid_col)

        # if grid_row >= 20 or grid_col >=20:
        #     print('stop',grid_row,grid_col,orig)
        return [grid_row, grid_col]

    def in_field_of_view(self, agent_id, obs):
        # obs = ('agent',agent_id) or ('item',item_id)
        idx = obs[1]
        if obs[0] == "agent":
            grid_cells = self.get_perimeter_grid_cells("agent", idx, self.agents[idx])
        elif obs[0] == "item":
            grid_cells = self.get_perimeter_grid_cells(
                "item", idx, self.item_states[idx]
            )

        entity_is_observable = [
            self.agents[agent_id].field_of_view[grid_cell[0], grid_cell[1]]
            for grid_cell in grid_cells
        ]
        return sum(entity_is_observable) > 0

    def particle_in_fov(self, agent_id, state, obs):
        # obs = ('agent',agent_id), ('item',item_id)
        self._setup_tmp(self.env_id, state)
        idx = obs[1]
        if obs[0] == "agent":
            grid_cells = self.get_perimeter_grid_cells(
                "agent", idx, self.tmp_agents[idx]
            )
        elif obs[0] == "item":
            grid_cells = self.get_perimeter_grid_cells(
                "item", idx, self.tmp_item_states[idx]
            )
        entity_is_observable = [
            self.agents[agent_id].field_of_view[grid_cell[0], grid_cell[1]]
            for grid_cell in grid_cells
        ]
        return sum(entity_is_observable) > 0

    def room_in_fov(self, agent_id, room_id):
        """return True if most of the room is visible for agent_id"""
        if room_id == 0 or room_id == 3:
            y1, y2 = 0, self.room_dim[1] // 2
        else:
            y1, y2 = self.room_dim[1] // 2, self.room_dim[1]
        if room_id == 0 or room_id == 1:
            x1, x2 = 0, self.room_dim[0] // 2
        else:
            x1, x2 = self.room_dim[0] // 2, self.room_dim[0]
        room = np.zeros(self.room_dim)
        room[x1:x2, y1:y2] = 1
        return (
            np.sum(self.agents[agent_id].field_of_view.astype(int) & room.astype(int))
            > 0
        )  # np.sum(room)//4

    # update observations
    def update_observations(self):
        # print('update_observations')
        for agent_id, agent in enumerate(self.agents):
            agent_shape = self.create_shape(agent_id, self.agent_states[agent_id])
            agent.last_observations = {}
            for other_agent_id, other_agent_state in enumerate(self.agent_states):
                other_agent_pos = other_agent_state["pos"]
                agent_grid_cells = self.get_perimeter_grid_cells(
                    "agent", other_agent_id, self.agents[other_agent_id]
                )
                agent_is_observable = [
                    agent.field_of_view[grid_cell[0], grid_cell[1]]
                    for grid_cell in agent_grid_cells
                ]
                other_agent_shape = self.create_shape(
                    other_agent_id, self.agent_states[other_agent_id], scale=1.2
                )
                # print(agent_shape)
                # print(other_agent_shape)
                # print(agent_id, 'intersects', other_agent_id, agent_shape.intersects(other_agent_shape))
                # print(agent_id, 'overlaps', other_agent_id, agent_shape.overlaps(other_agent_shape))
                # # plt.plot(*agent_shape.exterior.xy)
                # # plt.plot(*other_agent_shape.exterior.xy)
                # # plt.savefig('shapes.png')

                # perimeter in fov or self or attached to same item or colliding
                if (
                    sum(agent_is_observable) > 0
                    or other_agent_id == agent_id
                    or (
                        self.agent_states[agent_id]["attached"]
                        == self.agent_states[other_agent_id]["attached"]
                        is not None
                    )
                    or agent_shape.intersects(other_agent_shape)
                ):
                    agent.last_observations[("agent", other_agent_id)] = [
                        other_agent_pos
                    ]
                    # for other observed agent, if touching/attached to other item, add it in
                    if other_agent_id != agent_id:
                        for item_id in range(self.num_items):
                            if self.agent_states[other_agent_id][
                                "attached"
                            ] == item_id or other_agent_shape.intersects(
                                self.create_shape(
                                    self.num_agents + item_id,
                                    self.item_states[item_id],
                                    scale=1.2,
                                )
                            ):
                                agent.last_observations[("item", item_id)] = [
                                    self.item_states[item_id]["pos"]
                                ]

            for item_id, item_state in enumerate(self.item_states):
                item_grid_cells = self.get_perimeter_grid_cells(
                    "item", item_id, self.item_states[item_id]
                )
                item_is_observable = [
                    agent.field_of_view[grid_cell[0], grid_cell[1]]
                    for grid_cell in item_grid_cells
                ]
                item_shape = self.create_shape(
                    self.num_agents + item_id, self.item_states[item_id], scale=1.2
                )
                # perimeter in fov or item is touching or attached to agent
                if (
                    sum(item_is_observable) > 0
                    or agent_shape.intersects(item_shape)
                    or self.agent_states[agent_id]["attached"] == item_id
                ):
                    agent.last_observations[("item", item_id)] = [item_state["pos"]]
                    # if this item is attached/touching other agent or touching another item, add it in
                    for other_agent_id in set(range(self.num_agents)) - set([agent_id]):
                        if self.agent_states[other_agent_id][
                            "attached"
                        ] == item_id or item_shape.intersects(
                            self.create_shape(
                                other_agent_id,
                                self.agent_states[other_agent_id],
                                scale=1.2,
                            )
                        ):
                            agent.last_observations[("agent", other_agent_id)] = [
                                self.agent_states[other_agent_id]["pos"]
                            ]
                    for other_item_id in set(range(self.num_items)) - set([item_id]):
                        if item_shape.intersects(
                            self.create_shape(
                                self.num_agents + other_item_id,
                                self.item_states[other_item_id],
                                scale=1.2,
                            )
                        ):
                            agent.last_observations[("item", other_item_id)] = [
                                self.item_states[other_item_id]["pos"]
                            ]

    def update_observations_human_simulation(self):
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5
        # print('update_observations_human_simulation')
        for agent_id, agent in enumerate(self.agents):
            agent_state = self.agent_states[agent_id]
            agent_shape = self.create_shape(agent_id, agent_state)
            agent_pos = agent_state["pos"]
            agent.last_observations = {}
            # always add self
            agent.last_observations[("agent", agent_id)] = [agent_pos]
            # other agent touch
            other_agent_id = 1 - agent_id
            other_agent_shape = self.create_shape(
                other_agent_id, self.agent_states[other_agent_id], scale=1.2
            )
            touch_other_agent = agent_shape.intersects(other_agent_shape)
            # other agent attached to same item
            other_agent_state = self.agent_states[other_agent_id]
            attached_to_same_item = (
                agent_state["attached"] == other_agent_state["attached"] is not None
            )
            # see other agent perimeter
            other_agent_pos = other_agent_state["pos"]
            # agent_grid_cells = self.get_perimeter_grid_cells('agent', other_agent_id, self.agents[other_agent_id])
            # agent_grid_world_center = [[col_baseline+c,row_baseline-r] for r,c in zip(agent_grid_cells[:,0],agent_grid_cells[:,1])] #x,y
            # agent_is_observable = [self.is_visible(grid_cell,agent_id) and \
            #                        self.world_pos_in_cone(grid_cell, agent_id) for grid_cell in agent_grid_world_center]
            agent_is_observable = self.is_center_visible(
                other_agent_pos, agent_id
            ) and self.world_pos_in_cone(other_agent_pos, agent_id)
            # if touch_other_agent or attached_to_same_item or sum(agent_is_observable)>0:
            if touch_other_agent or attached_to_same_item or agent_is_observable:
                agent.last_observations[("agent", other_agent_id)] = [other_agent_pos]
                # for other observed agent, if touching/attached to other item, add it in
                for item_id in range(self.num_items):
                    if self.agent_states[other_agent_id][
                        "attached"
                    ] == item_id or other_agent_shape.intersects(
                        self.create_shape(
                            self.num_agents + item_id,
                            self.item_states[item_id],
                            scale=1.2,
                        )
                    ):
                        agent.last_observations[("item", item_id)] = [
                            self.item_states[item_id]["pos"]
                        ]

            for item_id, item_state in enumerate(self.item_states):
                # item_grid_cells = self.get_perimeter_grid_cells('item', item_id, self.item_states[item_id])
                # item_grid_world_center = [[col_baseline+c,row_baseline-r] for r,c in zip(item_grid_cells[:,0],item_grid_cells[:,1])] #x,y
                # item_is_observable = [self.is_visible(grid_cell,agent_id) and\
                #                       self.world_pos_in_cone(grid_cell, agent_id) for grid_cell in item_grid_world_center]
                item_is_observable = self.is_center_visible(
                    item_state["pos"], agent_id
                ) and self.world_pos_in_cone(item_state["pos"], agent_id)
                item_shape = self.create_shape(
                    self.num_agents + item_id, self.item_states[item_id], scale=1.2
                )
                # perimeter in fov or item is touching or attached to agent
                # if sum(item_is_observable) > 0 or \
                if (
                    item_is_observable
                    or agent_shape.intersects(item_shape)
                    or self.agent_states[agent_id]["attached"] == item_id
                ):
                    agent.last_observations[("item", item_id)] = [item_state["pos"]]
                    # if this item is attached/touching other agent or touching another item, add it in
                    for other_agent_id in set(range(self.num_agents)) - set([agent_id]):
                        if self.agent_states[other_agent_id][
                            "attached"
                        ] == item_id or item_shape.intersects(
                            self.create_shape(
                                other_agent_id,
                                self.agent_states[other_agent_id],
                                scale=1.2,
                            )
                        ):
                            agent.last_observations[("agent", other_agent_id)] = [
                                self.agent_states[other_agent_id]["pos"]
                            ]
                    for other_item_id in set(range(self.num_items)) - set([item_id]):
                        if item_shape.intersects(
                            self.create_shape(
                                self.num_agents + other_item_id,
                                self.item_states[other_item_id],
                                scale=1.2,
                            )
                        ):
                            agent.last_observations[("item", other_item_id)] = [
                                self.item_states[other_item_id]["pos"]
                            ]
            # print(agent_id,agent.last_observations)

    def get_subgoals_goto(self, entity_id, goal):
        # print('subgoals_goto:', entity_id, goal)
        if goal[0] == "LMA":
            if self.env_id == 0:
                if self.attached[entity_id]:
                    return [
                        "GE",
                        entity_id,
                        self.num_agents + self.agent_states[entity_id]["attached"],
                        -1,
                    ]
            if (
                self.get_reward_state(
                    entity_id, self.get_state(), "stop", None, None, goal
                )
                > -0.02
            ):
                if self.attached[entity_id]:
                    return [
                        "GE",
                        entity_id,
                        self.num_agents + self.agent_states[entity_id]["attached"],
                        -1,
                    ]
                else:
                    return ["stop"]
        if self.env_id == 0:  # when there are no obstacles
            return goal
        if goal[0] == "RO":
            entity_pos = self.item_states[goal[1]]["pos"]
        else:
            if entity_id < self.num_agents:
                entity_pos = self.agent_states[entity_id][
                    "pos"
                ]  # self.agents_pos[entity_id]
            else:
                entity_pos = self.item_states[entity_id - self.num_agents][
                    "pos"
                ]  # self.items_pos[entity_id - self.num_agents]
        entity_room = self._get_room_id(entity_pos)
        if goal[0] == "LMA":
            goal_room = self._get_room_id(self.landmark_centers[goal[2]])
        elif goal[0] in ["RA", "RO"]:
            goal_room = goal[2]
        else:  # TE
            if goal[2] >= self.num_agents:
                goal_room = self._get_room_id(
                    self.item_states[goal[2] - self.num_agents]["pos"]
                )
            else:
                goal_room = self._get_room_id(self.agent_states[1 - entity_id]["pos"])
        # print('entity room, goal room', entity_room, goal_room)
        if entity_room == goal_room:
            return goal
        elif goal[0] in ["TE", "GE"]:
            if goal[2] >= self.num_agents:
                item_id = goal[2] - 2
                if self.in_field_of_view(entity_id, ("item", item_id)):
                    return goal
            else:
                agent_id = goal[2]
                if self.in_field_of_view(entity_id, ("agent", agent_id)):
                    return goal

        path = self.path[(entity_room, goal_room)]
        if path is None:
            return goal
        else:
            if goal[0] in ["RO", "LMO"]:
                return ["RO", goal[1], path[0], +1]
                # return ['RO', entity_id-self.num_agents, goal_room, +1] #['RO', entity_id-self.num_agents, entity_room, +1] #TODO
            else:
                # return ['RA', entity_id, entity_room, +1]
                return [
                    "RA",
                    entity_id,
                    path[0],
                    +1,
                ]  # ['RA', entity_id, entity_room, +1] #TODO changed for stealing, make sure consistent with other scenarios

    def get_subgoals_avoid(self, agent_id):
        dist = self._get_dist_pos(
            self.agent_states[agent_id]["pos"], self.agent_states[1 - agent_id]["pos"]
        )
        print(
            "avoidance subgoal:",
            dist,
            1.5 * SIZE[self.sizes[1 - agent_id]] + 1.5 * SIZE[self.sizes[agent_id]],
        )
        if (
            dist
            < 1.5 * SIZE[self.sizes[1 - agent_id]] + 1.5 * SIZE[self.sizes[agent_id]]
        ):
            return ["TE", agent_id, 1 - agent_id, -1]
        else:
            return ["stop"]

    def get_subgoals_goto_help(self, agent_id, goal):
        # print('subgoals_goto_help:', agent_id, goal)
        # input('press any key to continue...')
        goal_agent = goal[1]
        if self.env_id == 0 and goal[0] == "LMA":
            return ["stop"]
        entity_pos = self.agent_states[goal_agent]["pos"]
        entity_room = self._get_room_id(entity_pos)
        agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
        if goal[0] == "LMA":
            item_room = self._get_room_id(self.item_states[0]["pos"])  # TODO item0
            goal_room = self._get_room_id(self.landmark_centers[goal[2]])
            if entity_room == goal_room:
                if item_room == agent_room and item_room == goal_room:
                    return goal
                else:
                    return ["stop"]
            path = self.path[(entity_room, goal_room)]
            if path is None:
                return goal
            elif goal[0] == "LMA":  ### TODO: more than 1 object
                doors = self.doors[(entity_room, goal_room)]
                blocked = False
                for door_id in doors:
                    if (
                        _get_dist(self.doors_pos[door_id], self.item_states[0]["pos"])
                        < SIZE[self.sizes[2]] + SIZE[self.sizes[goal[1]]]
                    ):  # TODO assumes item 0 is blocking
                        blocked = True
                        break
                # print('blocked:', blocked)
                if blocked:
                    # print(item_room, agent_room, entity_room, goal_room)
                    # input('press any key to continue...')
                    return self.get_subgoals_put(
                        agent_id, ["LMO", 0, item_room, +1]
                    )  # TODO assumes item 0 is blocking
                    # if item_room == agent_room:
                    #     return item_room * 3 + 2
                    # else:
                    #     path = self.path[(agent_room, item_room)]
                    #     if path is None:
                    #         return 58
                    #     else:
                    #         return 30 + path[0] * 3 + agent_id
                else:
                    if self.attached[agent_id] is not None:
                        return [
                            "GE",
                            agent_id,
                            self.num_agents + self.agent_state[agent_id]["attached"],
                            -1,
                        ]  # detach
                    else:
                        return ["stop"]
        elif goal[0] == "TE":  # touch entity
            if goal[2] >= self.num_agents:
                item_room = self._get_room_id(
                    self.item_states[goal[2] - self.num_agents]["pos"]
                )
                print("gotohelp:", entity_room, agent_room, item_room)
                if (
                    entity_room == item_room
                    and self.get_reward_state(
                        goal[1], self.get_state(), "stop", None, None, goal
                    )
                    > -0.1
                ):
                    if self.attached[agent_id] is not None:
                        return ["GE", agent_id, goal[2], -1]  # release
                    else:
                        return ["stop"]  # stop
                # goal_room = self._get_room_id(self.items_pos[0])
                if (
                    self.attached[agent_id] is not None
                ):  # or self._get_dist_pos(self.agents_pos[agent_id], self.items_pos[0]) < SIZE[self.sizes[agent_id]] + SIZE[self.sizes[2]] + 0.2:
                    if agent_room == entity_room:  # in the same room
                        return goal
                    else:
                        return self.get_subgoals_goto(
                            goal[2], ["RO", goal[2] - self.num_agents, entity_room, +1]
                        )  # approach the other agent
                else:
                    if (
                        self.once_attached[agent_id] and agent_room == item_room
                    ):  # TODO: more general
                        return self.get_subgoals_goto(
                            goal[2], ["RO", goal[2] - self.num_agents, entity_room, +1]
                        )
                    else:
                        return self.get_subgoals_put(
                            agent_id,
                            ["LMO", goal[2] - self.num_agents, entity_room, +1],
                        )  # get the item first
            else:
                if (
                    self.get_reward_state(
                        goal[1], self.get_state(), "stop", None, None, goal
                    )
                    > -0.1
                ):
                    if self.attached[agent_id] is not None:
                        return [
                            "GE",
                            agent_id,
                            self.num_agents + self.agent_state[agent_id]["attached"],
                            -1,
                        ]
                    # TODO added for informing, check that doesn't change other scenarios.
                    else:
                        return ["stop"]  # stop
                return self.get_subgoals_goto(
                    agent_id, ["TE", agent_id, 1 - agent_id, +1]
                )
        return goal

    def get_subgoals_goto_hinder(self, agent_id, goal, with_object=True):
        print(agent_id, "get_subgoals_goto_hinder", goal)
        goal_agent = goal[1]
        entity_pos = self.agent_states[goal_agent]["pos"]
        entity_room = self._get_room_id(entity_pos)
        agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
        if goal[0] == "LMA":
            item_room = self._get_room_id(self.item_states[0]["pos"])  # TODO 1 item
            if self.env_id == 0:
                return [goal[0], goal[1], goal[2], -1]
            goal_room = goal[2]
            if entity_room == goal_room:
                if self.attached[agent_id]:
                    return [
                        "GE",
                        agent_id,
                        self.num_agents + self.agent_states[agent_id]["attached"],
                        -1,
                    ]
                if agent_room != goal_room:
                    return self.get_subgoals_goto(
                        agent_id, ["LMA", agent_id, goal_room, +1]
                    )
                else:
                    return [goal[0], goal[1], goal[2], -1]
            path = self.path[(entity_room, goal_room)]
            all_path = [entity_room] + path
            doors = self.doors[(entity_room, goal_room)]
            blocked = self._is_blocked(doors, SIZE[self.sizes[goal_agent]])
            if blocked:
                return ["stop"]  # do not need to do anything
            # check if the object can block the path
            """TODO: when its own goal is putting an object to a location, it may not block with the object"""
            if with_object:
                best_room = None
                best_length = 10
                for room_id in range(len(path)):
                    door = doors[room_id]
                    room_id1, room_id2 = all_path[room_id], all_path[room_id + 1]
                    if self.doors_size[door] < 2 * SIZE[self.sizes[2]]:  # TODO 1 item
                        # print('goto_hinder, blocked door:', door, room_id1, room_id2, self.doors_size[door])
                        # TODO correct room_id?
                        if item_room == room_id1:
                            return self.get_subgoals_put(
                                agent_id, ["LMO", 0, room_id2, +1]
                            )  # TODO 1 item
                        if item_room == room_id2:
                            return self.get_subgoals_put(
                                agent_id, ["LMO", 0, room_id1, +1]
                            )  # TODO 1 item
                        path1 = self.path[(item_room, room_id1)]
                        path2 = self.path[(item_room, room_id2)]
                        if path1 is None and path2 is None:
                            continue
                        if path1 is not None:
                            if path2 is not None and len(path2) < len(path1):
                                cur_length = len(path1)
                                if cur_length < best_length:
                                    best_length = cur_length
                                    best_room = room_id1
                                # return self.get_subgoals_put(agent_id, room_id1 * 3 + 2)
                            else:
                                cur_length = len(path2)
                                if cur_length < best_length:
                                    best_length = cur_length
                                    best_room = room_id2
                                # return self.get_subgoals_put(agent_id, room_id2 * 3 + 2)
                        else:
                            cur_length = len(path1)
                            if cur_length < best_length:
                                best_length = cur_length
                                best_room = room_id1
                            # return self.get_subgoals_put(agent_id, room_id1 * 3 + 2)
                # print('goto_hinder, best_room:', best_room)
                if best_room is not None:  # check if it is too far
                    len2item = (
                        0
                        if agent_room == item_room or self.attached[agent_id]
                        else len(self.path[(agent_room, item_room)])
                    )
                    len2door = (
                        0
                        if item_room == best_room
                        else len(self.path[(item_room, best_room)])
                    )
                    # print('len:', len2item, len2door, len(path))
                    """TODO: use distance instead?"""
                    if len2item + len2door < len(path):
                        return self.get_subgoals_put(
                            agent_id, ["LMO", 0, best_room, +1]
                        )  # TODO 1 item

            # blocking with itself
            """TODO: add goto doors subgoals"""
            if self.attached[agent_id]:
                return [
                    "GE",
                    agent_id,
                    self.num_agents + self.agent_states[agent_id]["attached"],
                    -1,
                ]
            best_room = None
            shortest_path = 10
            for i, room_id in enumerate(all_path):
                if i < len(all_path) - 1:
                    if (
                        self.doors_size[self.connected[all_path[i]][all_path[i + 1]]]
                        >= 10 - 1e-6
                    ):  # if no wall at all
                        continue
                if agent_room == room_id:
                    continue  # return 12 + goal_id
                self_path = self.path[(agent_room, room_id)]
                if self_path is not None:
                    if best_room is None or len(self_path) < shortest_path:
                        best_room = room_id
                        shortest_path = len(self_path)
            if best_room is not None:
                return self.get_subgoals_goto(agent_id, ["RA", agent_id, best_room, +1])

        elif goal[0] == "TE":
            if goal[2] >= self.num_agents:
                item_id = goal[2] - self.num_agents
                item_room = self._get_room_id(self.item_states[item_id]["pos"])
            else:
                item_room = self._get_room_id(self.item_states[0]["pos"])  # TODO 1 item
                item_id = 0
            # print('hinder:', agent_id, goal)
            if self.env_id == 0:
                if not self.attached[agent_id]:
                    return self.get_subgoals_put(
                        agent_id, ["LMO", item_id, item_room, +1]
                    )
                else:
                    return ["TE", goal_agent, self.num_agents + item_id, -1]
            # if already not touchable
            if entity_room != item_room and self._is_blocked(
                self.doors[(entity_room, item_room)], SIZE[self.sizes[goal_agent]]
            ):
                if self.attached[agent_id]:
                    return [
                        "GE",
                        agent_id,
                        self.num_agents + self.agent_states[agent_id]["attached"],
                        -1,
                    ]
                else:
                    return ["stop"]

            if agent_room != item_room and self._is_blocked(
                self.doors[(agent_room, item_room)], SIZE[self.sizes[agent_id]]
            ):
                if self.attached[goal_agent] or entity_room == item_room:
                    return ["stop"]  # there is nothing it can do
                return self.get_subgoals_goto_hinder(
                    agent_id, ["RA", goal_agent, item_room, +1], with_object=False
                )  # try to block by itself
            if self.attached[goal_agent]:
                if not self.attached[agent_id]:
                    return self.get_subgoals_put(
                        agent_id, ["LMO", item_id, item_room, +1]
                    )
                else:
                    return ["TE", goal_agent, self.num_agents + item_id, -1]
            else:
                # hide in a room
                best_room = None
                best_path = 1e6
                for room_id in range(4):
                    if entity_room != room_id and item_room != room_id:
                        if (
                            self._is_blocked(
                                self.doors[(entity_room, room_id)],
                                SIZE[self.sizes[goal_agent]],
                            )
                            and not self._is_blocked(
                                self.doors[(item_room, room_id)],
                                SIZE[self.sizes[item_id]],
                            )
                            and (
                                agent_room == room_id
                                or not self._is_blocked(
                                    self.doors[(agent_room, room_id)],
                                    SIZE[self.sizes[agent_id]],
                                )
                            )
                        ):
                            path = self.path[(item_room, room_id)]
                            if best_room is None or len(path) < best_path:
                                best_room = room_id
                                best_path = len(path)
                if best_room is not None:
                    return self.get_subgoals_put(
                        agent_id, ["RO", item_id, best_room, +1]
                    )  # hind in a room
                if entity_room == item_room:
                    if not self.attached[agent_id]:
                        return self.get_subgoals_put(
                            agent_id, ["LMO", item_id, item_room, +1]
                        )
                    else:
                        return ["TE", goal_agent, self.num_agents + item_id, -1]

                path = self.path[(entity_room, item_room)]
                if len(path) > 1 and agent_room in path[:-1]:  # in the path of opponent
                    return self.get_subgoals_goto_hinder(
                        agent_id, ["RA", goal_agent, room_id, +1], with_object=False
                    )  # block by itself
                else:  # grab the object and go to other room or just avoid opponent
                    if not self.attached[agent_id]:
                        return self.get_subgoals_put(
                            agent_id, ["LMO", item_id, item_room, +1]
                        )
                    else:
                        best_dist = 0
                        best_room = None
                        for room_id in range(4):
                            if room_id != entity_room and room_id != item_room:
                                entity_dist = self._get_dist_room(
                                    self.agent_states[goal_agent]["pos"], room_id
                                )
                                item_dist = self._get_dist_room(
                                    self.item_states[item_id]["pos"], room_id
                                )
                                if item_dist < entity_dist and item_dist > best_dist:
                                    best_dist = item_dist
                                    best_room = room_id
                        if best_room is not None:
                            return self.get_subgoals_goto(
                                self.num_agents + item_id,
                                ["RO", item_id, best_room, +1],
                            )
                        return ["TE", goal_agent, self.num_agents + item_id, -1]

        return [goal[0], goal[1], goal[2], -1]

    def get_cost(self, agent_id, goal):
        if goal[0] == "LMA":
            # return self._get_dist_room(self.agent_states[goal[1]]['pos'], self.landmark_centers[goal[2]])
            return self._get_dist_pos(
                self.agent_states[goal[1]]["pos"], self.landmark_centers[goal[2]]
            )
        if goal[0] == "LMO":
            if self.agent_states[agent_id]["attached"] == goal[1]:
                return self._get_dist_pos(
                    self.item_states[goal[1]]["pos"], self.landmark_centers[goal[2]]
                )
            # return self._get_dist_pos(self.item_states[goal[1]]['pos'], self.landmark_centers[goal[2]]) + \
            return self._get_dist_pos(
                self.item_states[goal[1]]["pos"], self.agent_states[agent_id]["pos"]
            )
        return 0

    def get_subgoals_put(self, agent_id, goal):
        # print(agent_id,'get_subgoals_put',goal)
        if self.env_id == 0:
            if self.attached[agent_id] is None:
                if (
                    self.get_reward_state(
                        agent_id, self.get_state(), "stop", None, None, goal
                    )
                    > -0.05
                ):
                    return ["stop"]  # stop
                if self.check_attachable(agent_id, goal[1]):
                    return ["GE", agent_id, self.num_items + goal[1], +1]  # grab
                else:
                    return self.get_subgoals_goto(
                        agent_id, ["TE", agent_id, self.num_agents + goal[1], +1]
                    )  # approach the object
                # if not self.check_attachable(agent_id, goal[1], threshold=1.5):
                #     return self.get_subgoals_goto(agent_id, ['TE', agent_id, self.num_agents+goal[1], +1]) # approach the object
            elif (
                self.get_reward_state(
                    agent_id, self.get_state(), "stop", None, None, goal
                )
                > -0.05
                and not self.once_attached[agent_id]
            ):  # TODO: get rid of once_attached!
                return ["GE", agent_id, self.num_items + goal[1], -1]  # release
            return goal

        # give it to the other agent when blocked
        if self.attached[agent_id] is not None and (
            self.attached[1 - agent_id] is not None or self.num_agents < 2
        ):
            agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
            goal_room = goal[2]
            if agent_room != goal_room:
                path = self.path[(agent_room, goal_room)]
                doors = self.doors[(agent_room, goal_room)]
                blocked = self._is_blocked(doors, SIZE[self.sizes[agent_id]] * 1.5)
                if blocked:
                    return [
                        "GE",
                        agent_id,
                        self.num_agents + goal[1],
                        -1,
                    ]  # TODO not sure goal[1] has item

        if self.attached[agent_id] is None:
            if (
                self.get_reward_state(
                    agent_id, self.get_state(), "stop", None, None, goal
                )
                > -0.05
            ):
                return ["stop"]  # stop
            # print('get_subgoals_put:', self.check_attachable(agent_id, goal[1]))
            if self.check_attachable(agent_id, goal[1]):
                return ["GE", agent_id, self.num_items + goal[1], +1]  # grab
            else:
                return self.get_subgoals_goto(
                    agent_id, ["TE", agent_id, self.num_agents + goal[1], +1]
                )  # approach the object
            # if not self.check_attachable(agent_id, goal[1], threshold=1.5):
            #     return self.get_subgoals_goto(agent_id, ['TE', agent_id, self.num_agents+goal[1], +1]) # approach the object
            # if self._get_dist_pos(self.agent_states[agent_id]['pos'], self.item_states[goal[1]]['pos']) > SIZE[self.sizes[agent_id]] * 1.5 + SIZE[self.sizes[goal[1] + self.num_agents]] * 1.5:
            #      return self.get_subgoals_goto(agent_id, ['TE', agent_id, self.num_items+goal[1], +1]) # approach the object
        elif (
            self.get_reward_state(agent_id, self.get_state(), "stop", None, None, goal)
            > -0.05
            and not self.once_attached[agent_id]
        ):
            return ["GE", agent_id, self.num_items + goal[1], -1]  # release

        # moving to the goal place with the object
        agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
        item_room = self._get_room_id(self.item_states[goal[1]]["pos"])
        goal_room = goal[2]
        """TODO: CHECK THIS!!! may need to consider blocked or not blocked separately!"""
        if item_room == goal_room and agent_room == goal_room:
            return goal
        else:
            if (
                agent_room == goal_room
                and self.agent_states[agent_id]["attached"] != goal[1]
            ):
                path = self.path[(item_room, goal_room)]
                doors = self.doors[(item_room, goal_room)]
                # if self._is_blocked(doors, SIZE[self.sizes[2]]):
                return self.get_subgoals_goto(
                    agent_id, ["RO", goal[1], goal_room, +1]
                )  # TODO
            elif item_room == goal_room:
                path = self.path[(agent_room, goal_room)]
                doors = self.doors[(agent_room, goal_room)]
                # if self._is_blocked(doors, SIZE[self.sizes[agent_id]]):
                return goal
            else:
                return self.get_subgoals_goto(
                    agent_id, ["RO", goal[1], goal_room, +1]
                )  # TODO

        return self.get_subgoals_goto(agent_id, ["RO", goal[1], goal_room, +1])  # TODO

    def get_subgoals_put_help(self, agent_id, goal):
        # print(agent_id, 'get_subgoals_put_help', goal)
        """TODO: put entity to another entity"""
        if self.env_id == 0:
            return self.get_subgoals_put(agent_id, goal)

        if (
            self.get_reward_state(agent_id, self.get_state(), "stop", None, None, goal)
            > -0.05
        ):
            if self.attached[agent_id]:
                return [
                    "GE",
                    agent_id,
                    self.agent_states[agent_id]["attached"] + self.num_agents,
                    -1,
                ]  # 56 + agent_id
            else:
                return ["stop"]  # TODO: stop only once
        goal_agent = 1 - agent_id
        entity_pos = self.agent_states[goal_agent]["pos"]
        entity_room = self._get_room_id(entity_pos)
        item_room = self._get_room_id(
            self.item_states[goal[1]]["pos"]
        )  # for 2 items assuming LMO. previously - self.items_pos[0]
        agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
        goal_room = goal[2]

        print(
            self.attached[agent_id] is not None, self.attached[goal_agent] is not None
        )
        print(
            self.agent_states[agent_id]["attached"],
            self.agent_states[goal_agent]["attached"],
        )
        print(agent_room, goal_room)
        agent_goal_blocked = False
        if agent_room != goal_room:
            path = self.path[(agent_room, goal_room)]
            doors = self.doors[(agent_room, goal_room)]
            blocked = self._is_blocked(doors, SIZE[self.sizes[agent_id]] * 1.5)
            agent_goal_blocked = blocked
            print("blocked:", blocked)
        agent_item_blocked = False
        if agent_room != item_room:
            path = self.path[(agent_room, item_room)]
            doors = self.doors[(agent_room, item_room)]
            agent_item_blocked = self._is_blocked(
                doors, SIZE[self.sizes[agent_id]] * 1.5
            )
        entity_goal_blocked = False
        if entity_room != goal_room:
            path = self.path[(entity_room, goal_room)]
            doors = self.doors[(entity_room, goal_room)]
            entity_goal_blocked = self._is_blocked(
                doors, SIZE[self.sizes[goal_agent]] * 1.5
            )
        entity_item_blocked = False
        if entity_room != item_room:
            path = self.path[(entity_room, item_room)]
            doors = self.doors[(entity_room, item_room)]
            entity_item_blocked = self._is_blocked(
                doors, SIZE[self.sizes[goal_agent]] * 1.5
            )

        # can finish the whole task
        if not agent_item_blocked and not agent_goal_blocked:
            return self.get_subgoals_put(agent_id, goal)

        if (
            self.attached[agent_id] is not None
            and self.attached[goal_agent] is not None
        ):
            if agent_goal_blocked:
                return [
                    "GE",
                    agent_id,
                    self.agent_states[agent_id]["attached"] + self.num_agents,
                    -1,
                ]  # 56 + agent_id

        # TODO no need to help if goal_agent is in goal room, has item and door is blocked for agent_id
        # needed for after letting go, let heplee move away
        if self.attached[goal_agent] is not None:
            return ["stop"]
            # if agent_goal_blocked:
            #     return ['stop']

        # subgoal_opponent = self.get_subgoals_put(goal_agent, goal_id)
        """TODO: need a better way to determine obstacle"""

        if not entity_goal_blocked and entity_item_blocked:  # help getting the item
            if agent_item_blocked:
                return ["stop"]
            else:
                return self.get_subgoals_goto_help(
                    agent_id, ["TE", goal_agent, goal[1] + self.num_agents, +1]
                )  # 25 + goal_agent

        if entity_goal_blocked and not entity_item_blocked:  # help putting the item
            if agent_goal_blocked:
                return ["stop"]
            else:
                return self.get_subgoals_put(agent_id, goal)

        # if not self.once_attached[goal_agent]:
        #     if entity_item_blocked: # help get the object if blocked
        #         return self.get_subgoals_goto_help(agent_id, ['TE', goal_agent, goal[1]+self.num_agents, +1]) #25 + goal_agent
        #     else:
        #         return ['stop']
        # else:
        #     # can not help
        #     if agent_room != goal_room:
        #         path = self.path[(agent_room, goal_room)]
        #         doors = self.doors[(agent_room, goal_room)]
        #         blocked = False
        #         for door in doors:
        #             if self.doors_size[door] < 2 * SIZE[self.sizes[agent_id]]:
        #                 blocked = True
        #                 break
        #         if blocked:
        #             return ['stop']
        #     # whether need to help
        #     if goal_room != entity_room: # in the process of pushing the object
        #         path = self.path[(entity_room, goal_room)]
        #         doors = self.doors[(entity_room, goal_room)]
        #         blocked = False
        #         for door in doors:
        #             if self.doors_size[door] < 2 * SIZE[self.sizes[goal_agent]]:
        #                 blocked = True
        #                 break
        #         if blocked: # help push the object if blocked
        #             return self.get_subgoals_put(agent_id, goal)
        #         # else:
        #         #     return 58

        return goal

    def get_subgoals_put_hinder(self, agent_id, goal):
        # print(agent_id, 'get_subgoals_put_hinder', goal)
        # TODO: not accessible, if on landmark, 12 + goal_id, else (if attached: detach else: 58)

        # TODO why return same goal for env 0?
        if self.env_id == 0:
            return ["LMO", goal[1], goal[2], -1]
        goal_agent = 1 - agent_id
        entity_pos = self.agent_states[goal_agent]["pos"]
        entity_room = self._get_room_id(entity_pos)
        item_room = self._get_room_id(self.item_states[goal[1]]["pos"])
        agent_room = self._get_room_id(self.agent_states[agent_id]["pos"])
        goal_room = goal[2]

        if (
            self.env_id == 4
            and self.sizes[agent_id] == 0
            and self.sizes[self.num_agents + goal[1]] == 0
        ):
            if entity_room != item_room and self._is_blocked(
                self.doors[(entity_room, item_room)], SIZE[self.sizes[agent_id]]
            ):
                return ["stop"]  # do nothing
            return self.get_subgoals_put(agent_id, ["RO", goal[1], 0, +1])

        if agent_room != item_room and self._is_blocked(
            self.doors[(agent_room, item_room)], SIZE[self.sizes[agent_id]]
        ):
            if self.attached[goal_agent] or entity_room == item_room:
                return ["stop"]  # there is nothing it can do
            return self.get_subgoals_goto_hinder(
                agent_id, ["RA", goal_agent, item_room, +1], with_object=False
            )  # try to block by itself

        # # no need to hinder
        # if not self.once_attached[goal_agent]:
        #     if entity_room != item_room and self._is_blocked(self.doors[(entity_room, item_room)], SIZE[self.sizes[goal_agent]])
        #         """TODO: if item is already on landmark?"""
        #         if self.attached[agent_id]:
        #             return 56 + agent_id
        #         else:
        #             return 58
        if self.attached[goal_agent]:  # in the process of pushing the object
            # if item_room == goal_room:
            #     return 12 + goal_agent
            # check if achievable
            if (
                entity_room != goal_room
                and self._is_blocked(
                    self.doors[(entity_room, goal_room)], SIZE[self.sizes[goal_agent]]
                )
                or item_room != goal_room
                and self._is_blocked(
                    self.doors[(item_room, goal_room)], SIZE[self.sizes[goal_agent]]
                )
            ):
                if self.attached[agent_id]:
                    return [
                        "GE",
                        agent_id,
                        self.num_agents + self.agent_states[agent_id]["attached"],
                        -1,
                    ]
                else:
                    return ["stop"]

            if not self.attached[agent_id]:
                return self.get_subgoals_put(agent_id, ["LMO", goal[1], item_room, +1])
            # else:
            #     self.get_subgoals_goto_hinder(agent_id, goal_room * 3 + 2, with_object=False)
        # else:
        # no need to hinder -- opponent can not get the object
        if entity_room != item_room and self._is_blocked(
            self.doors[(entity_room, item_room)], SIZE[self.sizes[goal_agent]]
        ):
            # if self.attached[agent_id]:
            #     return 56 + agent_id
            # else:
            return ["stop"]

        # hide in a room
        for room_id in range(4):
            if entity_room != room_id and item_room != room_id:
                if (
                    self._is_blocked(
                        self.doors[(entity_room, room_id)], SIZE[self.sizes[goal_agent]]
                    )
                    and not self._is_blocked(
                        self.doors[(item_room, room_id)],
                        SIZE[self.sizes[self.num_agents + goal[1]]],
                    )
                    and (
                        agent_room == room_id
                        or not self._is_blocked(
                            self.doors[(agent_room, room_id)],
                            SIZE[self.sizes[agent_id]],
                        )
                    )
                ):
                    return self.get_subgoals_put(
                        agent_id, ["RO", goal[1], room_id, +1]
                    )  # hide in a room
        # if entity_room == item_room:
        #     if not self.attached[agent_id]:
        #         return self.get_subgoals_put(agent_id, item_room * 3 + 2)
        #     else:
        #         return 12 + goal_agent

        if entity_room != item_room:
            path = self.path[(entity_room, item_room)]
            if len(path) > 1 and agent_room in path[:-1]:  # in the path of opponent
                return self.get_subgoals_goto_hinder(
                    agent_id, ["RA", goal_agent, room_id, +1], with_object=False
                )  # block by itself
        # grab the object and go to other room or just avoid opponent
        if not self.attached[agent_id]:
            return ["GE", agent_id, goal[1] + self.num_agents, +1]
            # return self.get_subgoals_put(agent_id, ['LMO', goal[1], item_room, +1]) #TODO isn't this helping?!
        else:
            best_dist = 0
            best_room = None
            for room_id in range(4):
                if room_id != entity_room and room_id != item_room:
                    entity_dist = self._get_dist_room(
                        self.agent_states[goal_agent]["pos"], room_id
                    )
                    item_dist = self._get_dist_room(
                        self.item_states[goal[1]]["pos"], room_id
                    )
                    if item_dist < entity_dist and item_dist > best_dist:
                        best_dist = item_dist
                        best_room = room_id
            if best_room is not None:
                return self.get_subgoals_goto(
                    self.num_agents + goal[1], ["RO", goal[1], best_room, +1]
                )
            return [
                "LMA",
                goal_agent,
                entity_room,
                -1,
            ]  # ['LMA', goal_agent, 0, -1] #TODO why 0 and not remove from entity_room / item_room?

        return [
            "LMA",
            goal_agent,
            entity_room,
            -1,
        ]  # ['LMA', goal_agent, 0, -1] #TODO why 0?

    def multiple_goals(self, goals, agent_id):
        goal1, goal2 = goals
        goal1_item = goal1[1]
        goal1_landmark = goal1[2]
        # print('landmark dist',_get_dist(self.item_states[goal1_item]['pos'], self.landmark_centers[goal1_landmark]))
        # goal1 - goal1 item not on target landmark: detach other items, get goal1 item
        if (
            _get_dist(
                self.item_states[goal1_item]["pos"],
                self.landmark_centers[goal1_landmark],
            )
            > 2.5
        ):
            attached = self.agent_states[agent_id]["attached"]
            # print('landmark dist attached',attached)
            if attached is not None and attached != goal1_item:
                return ["GE", agent_id, self.num_agents + attached, -1]
            return self.get_subgoals_put(agent_id, goal1)
        # defend - goal1 item and other agent on target landmark room (landmark#=room#)
        elif (
            self._get_room_id(self.agent_states[1 - agent_id]["pos"])
            == self._get_room_id(self.item_states[goal1_item]["pos"])
            == goal1_landmark
        ):
            return [
                "LMA",
                1 - agent_id,
                goal1_landmark,
                -1,
            ]  # TODO prevent touching - if actually see green call goto_hinder decompos + change all particle decom - if exists 1 goto
        # goal2
        else:
            if self.num_agents == 2:
                if self.agent_states[1 - agent_id]["attached"] == goal1[1]:
                    print("opponent grabbed")
                    return self.get_subgoals_put(agent_id, goal1)
                room_id = self._get_room_id(self.agent_states[1 - agent_id]["pos"])
                """TODO: also check visbility -- if visible then prevent it from reaching the room"""
                visible = self.in_field_of_view(agent_id, ("agent", 1 - agent_id))
                print("visbility", visible)
                if room_id == goal1[2] or visible:
                    if visible:
                        return self.get_subgoals_goto_hinder(
                            agent_id,
                            ["LMA", 1 - agent_id, room_id, +1],
                            with_object=False,
                        )
                    else:
                        return self.get_subgoals_goto(
                            agent_id, ["TE", agent_id, 1 - agent_id, +1]
                        )
            print("object 2")
            return self.get_subgoals_put(agent_id, goal2)
