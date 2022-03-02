from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def bfs(connected, start_pos, end_pos):
    """shortest path search givn connectivity between rooms"""
    N = len(connected)
    q = [start_pos]
    pre = {start_pos: None}
    l, r = -1, 0
    found = False
    while not found and l < r:
        l += 1
        cur_pos = q[l]
        for nxt_pos in range(N):
            if connected[cur_pos][nxt_pos] is not None and nxt_pos not in q:
                pre[nxt_pos] = cur_pos
                if nxt_pos == end_pos:
                    found = True
                    break
                q.append(nxt_pos)
                r += 1

    if not found:
        return None
    cur_pos = end_pos
    path = []
    doors = []
    while cur_pos != start_pos:
        path = [cur_pos] + path
        doors = [connected[pre[cur_pos]][cur_pos]] + doors
        cur_pos = pre[cur_pos]
    return path, doors


def get_ave_vel(trajs):
    ave_vel_list = []
    for traj in trajs:
        ave_vel = 0
        for t in range(1, len(traj)):
            cur_vel = ((traj[t][0] - traj[t - 1][0]) ** 2 + (traj[t][0] - traj[t - 1][0]) ** 2) ** 0.5
            ave_vel += cur_vel
        if len(traj) > 1:
            ave_vel /= len(traj) - 1
        ave_vel_list.append(ave_vel)
    return ave_vel_list


def get_dist(pos1, pos2):
    """get the distance between two 2D positions"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def weights_init(m):
    """
    initializaing weights
    """
    initrange = 0.1
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)


def update_network(loss, optimizer):
    """update network parameters"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """load trained model parameters"""
    model.load_state_dict(dict(torch.load(path)))


def to_variable(tensor, device):
    return tensor.to(device)


def one_hot(x, n):
    return [1 if i == x else 0 for i in range(n)]


def get_node_inputs(trajectories, sizes, num_agents, landmark_centers, wall_segs, dT=5):
    """
    convert trajectories and the environment layout into graph nodes
    type (4)
    size (1)
    coord (2)
    vel (2)
    orientation (1)
    """
    T = len(trajectories[0])
    num_entities = len(trajectories)
    num_landmarks = len(landmark_centers)
    N = num_entities + num_landmarks + len(wall_segs)
    t_list = list(range(dT, T, dT))
    num_steps = len(t_list)
    node_inputs = [None] * N
    for entity_id, trajs in enumerate(trajectories[:-1]):
        entity_type = 0 if entity_id < num_agents else 1 + entity_id - num_agents
        entity_one_hot = one_hot(entity_type, 8)
        size = sizes[entity_id]
        current_node_input = [None] * num_steps
        for step, t in enumerate(t_list):
            current_node_input[step] = list(entity_one_hot) + [size] + list(trajs[t])
        node_inputs[entity_id] = current_node_input
    
    for landmark_id, landmark_center in enumerate(landmark_centers):
        entity_one_hot = one_hot(3 + landmark_id, 8)
        size = 2.5
        current_node_input = [None] * num_steps
        for step in range(num_steps):
            current_node_input[step] = list(entity_one_hot) + [size] + list(landmark_center) + [0, 0, 0]
        node_inputs[num_entities + landmark_id] = current_node_input

    for wall_id, wall_seg in enumerate(wall_segs):
        entity_one_hot = one_hot(7, 8)
        size = (abs(wall_seg[0][0] - wall_seg[1][0]) + abs(wall_seg[0][1] - wall_seg[1][1])) * 0.5
        coord = ((wall_seg[0][0] + wall_seg[1][0]) * 0.5, (wall_seg[0][1] + wall_seg[1][1]) * 0.5)
        current_node_input = [None] * num_steps
        for step in range(num_steps):
            current_node_input[step] = list(entity_one_hot) + [size] + list(coord) + [0, 0, 0]
        node_inputs[num_entities + num_landmarks + wall_id] = current_node_input

    return np.array(node_inputs)


def get_goal_code(goal):
    """get goal code"""
    if goal == 'help': # helping (total: 1)
        return 15
    elif goal == 'hinder': # hindering (total: 1)
        return 16
    elif goal[-1] > 0:
        if goal[0] == 'LMA': # self to landmark (total: 4)
            return goal[2]
        elif goal[0] == 'TE': # self to entity (total: 3)
            if goal[2] < 2:
                return 4
            else:
                return 4 + goal[2] - 1
        elif goal[0] == 'LMO': # object to landmark (total: 8)
            return 7 + goal[1] * 4 + goal[2]
        elif goal[0] == 'help': # helping (total: 1)
            return 15
        elif goal[0] == 'hinder': # hindering (total: 1)
            return 16
        else:
            raise ValueError('Invalid goal:', goal)
    else:
        if goal[0] == 'TE' and goal[1] < 2 and goal[2] < 2: # avoiding (total: 1)
            return 17
        else:
            raise ValueError('Invalid goal:', goal)


def get_goal_GT(goals):
    """
    ground-truth goals (single goal)
    """
    return np.array([get_goal_code(goal) for goal in goals])







