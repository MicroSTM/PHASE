import numpy as np
import json

class MazeSampler():
    """
    sample symbolic representations of maze environments
    1-4 rooms
    """

    MAZE_TYPES = [0, 1, 2, 3, 4]
    WALL_TYPES = ['..........', 
                 '.....=====', '=====.....',
                 '...=======', '=======...',
                 # '....====', '==....==', '====....',
                 # '..======', '===..===', '======..',
                 '==========']


    def __init__(self, 
                 width=10, 
                 height=10):
        self.width = width
        self.height = height
        self.num_envs = 0
        self.env_defs = {}


    def dump(self, path):
        json.dump(self.env_defs, open(path, 'w'))


    def load(self, path):
        self.env_defs = json.load(open(path, 'r'))
        self.num_envs = len(self.env_defs.keys())


    def env_def(self, env_id):
        return self.env_defs[env_id]


    def gen_env_defs(self):
        self.num_envs = 0
        counts = {maze_type: 0 for maze_type in self.MAZE_TYPES}
        maze_list = self._create_maze()
        for maze_def in maze_list:
            env_def = {
            'maze_type': maze_def['maze_type'],
            'maze_def': maze_def['maze_def'],
            }
            # print(env_def)
            self.env_defs[self.num_envs] = env_def
            self.num_envs += 1
            counts[maze_def['maze_type']] += 1
        # print(counts)


    def get_env_def(self, env_id):
        return self.env_defs[env_id]


    def _create_maze(self):
        """TODO: check similarity"""
        maze_list = []
        for wall_type_4 in self.WALL_TYPES:
            for wall_type_3 in self.WALL_TYPES:
                for wall_type_2 in self.WALL_TYPES:
                    for wall_type_1 in self.WALL_TYPES:
                        maze_def = [wall_type_1, wall_type_2, wall_type_3, wall_type_4]
                        num_walls = sum([int(wall_type != '..........') for wall_type in maze_def])
                        if not _valid(maze_def, num_walls): continue
                        if num_walls == 1: continue
                        if _check_unique(maze_list, maze_def):
                            maze_list.append({'maze_type': num_walls, 'maze_def': maze_def})
        return maze_list


def _valid(maze_def, num_walls):
    if num_walls > 2 and sum([int(wall_type.startswith('.') and wall_type != '..........') for wall_type in maze_def]) > 0: return False
    if sum([int(wall_type == '==========') for wall_type in maze_def]) > 1: return False
    if sum([int(wall_type == '==========') for wall_type in maze_def]) == 0 and num_walls > 0: return False
    # if num_walls > 2 and (sum([int(wall_type in ['....====', '====....']) for wall_type in maze_def]) == 0 or 
    #                       sum([int(wall_type in ['..======', '======..']) for wall_type in maze_def]) == 0): return False
    return True


def _check_unique(maze_list, maze_def):
    if not maze_list:
        return True
    for env_def in maze_list:
        def_ = env_def['maze_def']
        if _same(def_, maze_def): return False
        tmp = def_
        for rotate in range(3):
            tmp = _rotate(tmp)
            if _same(tmp, maze_def): return False
        if _same(_flipping_x(def_), maze_def): return False
        if _same(_flipping_y(def_), maze_def): return False
        if _same(_flipping_diag_1(def_), maze_def): return False
        if _same(_flipping_diag_2(def_), maze_def): return False
    return True


def _same(def1, def2):
    for d1, d2 in zip(def1, def2):
        if d1 != d2:
            return False
    return True


def _rotate(maze_def):
    new_def = list(maze_def)
    for i in range(4):
        new_def[(i + 1) % 4] = maze_def[i]
    return new_def


def _flipping_x(maze_def):
    new_def = list(maze_def)
    new_def[1] = maze_def[3]
    new_def[3] = maze_def[1]
    return new_def


def _flipping_y(maze_def):
    new_def = list(maze_def)
    new_def[0] = maze_def[2]
    new_def[2] = maze_def[0]
    return new_def


def _flipping_diag_1(maze_def):
    new_def = list(maze_def)
    new_def[0] = maze_def[1]
    new_def[1] = maze_def[0]
    new_def[2] = maze_def[3]
    new_def[3] = maze_def[2]
    return new_def
    

def _flipping_diag_2(maze_def):
    new_def = list(maze_def)
    new_def[0] = maze_def[3]
    new_def[3] = maze_def[0]
    new_def[2] = maze_def[1]
    new_def[1] = maze_def[2]
    return new_def
