import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely import affinity
import math
import matplotlib.pyplot as plt

from utils import *


#TODO can add fov in here as polygon?
class GeometricEnv:
    def __init__(self, room_bound, room_center, doors_pos_start, sizes, num_agents, num_items):
        min_x, min_y, max_x, max_y = room_bound  # bottom-left corner + upper-right corner
        c_x, c_y = room_center
        w1, w2, w3, w4 = doors_pos_start
        self.sizes = sizes
        self.num_agents = num_agents
        self.num_items = num_items
        self.entities = {'agents':[], 'items':[]}
        self.room = Polygon([(min_x,min_y), (min_x,max_y), (max_x,max_y), (max_x,min_y),(min_x,min_y)])
        self.maze_walls = [LineString([(c_x, c_y), (w1[0],w1[1])]),
                           LineString([(c_x, c_y), (w2[0],w2[1])]),
                           LineString([(c_x, c_y), (w3[0],w3[1])]),
                           LineString([(c_x, c_y), (w4[0],w4[1])])]


    def set_state(self, state):
        self.entities = {'agents':[], 'items':[]}
        for agent_id, agent in enumerate(state[:self.num_agents]):
            polygon = create_agent_shape(agent['pos'], agent['angle'], self.sizes[agent_id])
            self.entities['agents'].append(polygon)
        for item_id,item in enumerate(state[self.num_agents:]):
            circle = create_item_shape(item['pos'], self.sizes[self.num_agents+item_id])
            self.entities['items'].append(circle)

        # colors = ['r','lime','c','pink','black']
        # for e, c in zip(self.entities['agents']+self.entities['items']+[self.room], colors):
        #     plt.plot(*e.exterior.xy,c)
        # for w in self.maze_walls:
        #     plt.plot(*w.xy,'black')
        # plt.show()


    def detect_entity_collision(self, entity_type, entity_id):
        """with other entities"""
        entity = self.entities[entity_type][entity_id]
        collisions = []
        for agent_id,agent in enumerate(self.entities['agents']):
            if entity.touches(agent):
                collisions.append(('agent',agent_id))
        for item_id,item in enumerate(self.entities['items']):
            if entity.touches(item):
                collisions.append(('item',item_id))
        return collisions


    def detect_overlap(self, entity_type, entity_id):
        """with entities and maze walls"""
        #intersection that is not just touches
        entity = self.entities[entity_type][entity_id]
        overlap = []
        for agent_id,agent in enumerate(self.entities['agents']):
            if entity.intersects(agent) and not entity.touches(agent):
                overlap.append(('agent',agent_id))
        for item_id,item in enumerate(self.entities['items']):
            if entity.intersects(item) and not entity.touches(item):
                overlap.append(('item',item_id))
        for wall_id,wall in enumerate(self.maze_walls):
            if entity.intersects(wall) and not entity.touches(wall):
                overlap.append(('wall',wall_id))
        return list(set(overlap)-set([(entity_type[:-1], entity_id)]))


    def in_room(self, entity_type, entity_id):
        entity = self.entities[entity_type][entity_id]
        return self.room.contains(entity)
