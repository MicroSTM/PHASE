# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import timeit
import math
import pickle
import numpy as np
import random

from agents.MCTS_particle_2agents import MCTS_particle_2agents

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123, help="Random seed")
parser.add_argument(
    "--t-max",
    type=int,
    default=40,
    metavar="STEPS",
    help="Max number of forward steps for A2C before update",
)
parser.add_argument(
    "--max-episode-length",
    type=int,
    default=35,
    metavar="LENGTH",
    help="Maximum episode length",
)
parser.add_argument(
    "--hidden-size",
    type=int,
    default=128,
    metavar="SIZE",
    help="Hidden size of LSTM cell",
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints",
    metavar="RECORD",
    help="Mission record directory",
)
parser.add_argument(
    "--record-dir",
    type=str,
    default="record",
    metavar="RECORD",
    help="Mission record directory",
)
parser.add_argument("--exp-name", type=str, default="Maze_v1", help="Experiment name")
parser.add_argument(
    "--checkpoints",
    nargs="*",
    type=int,
    default=[0, 0],
    help="checkpoints to be loaded",
)
parser.add_argument("--planning-agent", type=int, default=0, help="Planning agent")
parser.add_argument("--action_freq", type=int, default=1, help="Action frequency")
parser.add_argument("--verbose", type=int, default=1, help="How much info to display")
parser.add_argument("--env-id", type=int, default=13, help="Environment id")
parser.add_argument(
    "--goals", nargs="*", default=[None, None], help="Goals of the agents"
)  # [41,35],[11,55], [62,26]protecting, [11,26] put in room/touch
parser.add_argument("--goal1", nargs="*", default=[], help="Goals of the agents")
parser.add_argument("--goal2", nargs="*", default=[], help="Goals of the agents")
parser.add_argument(
    "--strengths", nargs="*", type=int, default=[3, 1], help="Strengths of the agents"
)
parser.add_argument(
    "--sizes", nargs="*", type=int, default=[0, 0, 0, 0], help="Sizes of the entities"
)
parser.add_argument(
    "--densities",
    nargs="*",
    type=int,
    default=[0, 0, 0, 0],
    help="Densities of the entities",
)
parser.add_argument(
    "--init-positions",
    nargs="*",
    type=int,
    default=[7, 3, 10, 4],
    help="initial positions of the entities",
)
parser.add_argument(
    "--init-agent-angles",
    nargs="*",
    type=float,
    default=[0, -math.pi],
    help="initial angles of the agents",
)
parser.add_argument(
    "--costs", nargs="*", type=int, default=[0, 0], help="Costs of the agents"
)
parser.add_argument(
    "--levels", nargs="*", type=int, default=[0, 0], help="Levels of the agents"
)
parser.add_argument(
    "--max-nb-episodes",
    type=int,
    default=10,
    help="Maximum number of planning episodes",
)
parser.add_argument(
    "--enable-renderer",
    action="store_true",
    default=True,
    help="True - render display; False - disable rendering",
)
parser.add_argument(
    "--enable-belief-renderer",
    action="store_true",
    default=False,
    help="True - render belief display; False - disable belief rendering",
)
parser.add_argument(
    "--balanced-sample",
    action="store_true",
    default=False,
    help="True - using pos-neg balanced samples for training",
)
parser.add_argument(
    "--input-type", type=str, default="image", help="Input type: image or state"
)
parser.add_argument(
    "--increase-density",
    action="store_true",
    default=False,
    help="Whether to increase the density of the item",
)
parser.add_argument("--cInit", type=float, default=1.25, help="Hyper-param for MCTS")
parser.add_argument("--cBase", type=float, default=1000, help="Hyper-param for MCTS")
parser.add_argument(
    "--nb-simulations", type=int, default=1000, help="Number of MCTS simulations"
)  # 1000
parser.add_argument(
    "--max-rollout-steps",
    type=int,
    default=10,
    help="Maximum number of rollout steps in a simulation",
)
parser.add_argument("--nb-samples", type=int, default=1, help="Number of samples")
parser.add_argument(
    "--nb-processes", type=int, default=1, help="Number of parallel processes"
)
parser.add_argument(
    "--action-type",
    type=str,
    default="force",
    help="force - applying forces; impulse - applying impulses",
)
parser.add_argument(
    "--temporal-decay",
    nargs="*",
    type=int,
    default=[0, 0],
    help="temporal decay of belief certainty",
)
parser.add_argument(
    "--visibility",
    nargs="*",
    type=int,
    default=[1, 1, 1, 1],
    help="0 - not rendered, 1 - rendered",
)
parser.add_argument(
    "--action-space-types",
    nargs="*",
    type=int,
    default=[0, 0],
    help="0 - full action space, 1 - not detach, 2 - no attach/detach",
)
parser.add_argument(
    "--random-colors",
    action="store_true",
    default=False,
    help="True - randomize entities' colors, False - fix entities' colors",
)
parser.add_argument(
    "--random-colors-agents",
    action="store_true",
    default=False,
    help="True - randomize agents' colors, False - fix all entities' colors",
)
parser.add_argument(
    "--init-times", nargs="*", type=int, default=[0, 0], help="Initial time periods"
)
parser.add_argument(
    "--init-goals", nargs="*", type=int, default=[58, 58], help="initial goals"
)
parser.add_argument("--plan-length", type=int, default=1, help="Length of plans")
parser.add_argument(
    "--execute-length", type=int, default=1, help="Length of plans that get executed"
)
parser.add_argument(
    "--alpha", nargs="*", type=float, default=[0.0, 0.0], help="Relation factors"
)
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
parser.add_argument("--num-items", type=int, default=2, help="Number of items")
parser.add_argument(
    "--all-directions",
    action="store_true",
    default=False,
    help="True - moving in all directions; False - only moving in gaze direction",
)
parser.add_argument(
    "--full-obs",
    nargs="*",
    type=int,
    default=[0, 0],
    help="1 - full obs, 0 - partial obs",
)
parser.add_argument(
    "--n-particles", type=int, default=50, help="Number of belief particles"
)
parser.add_argument(
    "--belief-prior", default=0.5, help="Over other agent's belief accuracy"
)
parser.add_argument(
    "--simulate-n-steps", default=5, help="Simulation horizon T for plan evaluation"
)
parser.add_argument(
    "--simulate-score-thres",
    default=1.5,
    help="When a plan score is below this thres, resample particles",
)
# prob - 0.05 without annealing, smaller with beta.
# state dist - 5, 2, 1
parser.add_argument("--beta", default=0.2, help="Plan score annealing")
parser.add_argument(
    "--save-date",
    action="store_true",
    default=False,
    help="True - save with date (for replay, keep original date); False - only planning params",
)
parser.add_argument(
    "--save-particles-separately",
    action="store_true",
    default=False,
    help="True - render all particles separately; False - only render all particles",
)
parser.add_argument(
    "--use-all-init-grid-pos",
    action="store_true",
    default=False,
    help="True - all grid init pos; False - limited POS set",
)
parser.add_argument(
    "--replay-name", type=str, default="independent", help="Experiment name"
)
parser.add_argument(
    "--replay-dir",
    type=str,
    default="record/replay",
    metavar="RECORD",
    help="Mission record directory",
)

# parser.add_argument('--flip-entity-colors', action='store_true', default=False, help='True - flip agent and or item colors, run replay; False - same entity colors')
parser.add_argument(
    "--flip-env-id",
    action="store_true",
    default=False,
    help="True - change env to env with similar structure and plan again with same initial params; False - same env",
)
parser.add_argument(
    "--flip-st-sz",
    action="store_true",
    default=False,
    help="True - change strength and size with same delta and plan again; False - same strength and size",
)
# parser.add_argument('--flip-env', type=int, default=None, help='0:up-down, 1:left-right, 2:diag1 y=x, 3:diag2 y=-x')
parser.add_argument(
    "--flip-env",
    action="store_true",
    default=None,
    help="0:up-down, 1:left-right, 2:diag1 y=x, 3:diag2 y=-x",
)
# parser.add_argument('--rotate-env', action='store_true', default=False, help='True - rotate 90 clockwise, False - no rotation')
parser.add_argument(
    "--flip-file-name",
    action="store_true",
    default=False,
    help="flip goals and entity order based on color and maze flip",
)


if __name__ == "__main__":
    args = parser.parse_args()

    ##############################
    # args.replay_name = ""
    # args.flip_entity_colors = True
    # args.rotate_env = True
    # args.save_date = True
    ##############################

    args.replay_dir = (
        args.record_dir + "/" + args.replay_name + "/replay"
    )  # save output
    parameters_list = pickle.load(
        open(args.record_dir + "/" + args.replay_name + "/parameters.pik", "rb")
    )  # create in get_parameters.py
    print(args.record_dir + "/" + args.replay_name + "/parameters.pik")
    print(parameters_list)
    for episode_id, parameters in enumerate(parameters_list):
        # for episode_id, parameters in enumerate(parameters_list[:2]):
        arg, path = parameters[0], parameters[1]
        args.env_id = arg["env_id"]
        args.goals = arg["goals"]
        args.strengths = arg["strengths"]
        args.sizes = arg["sizes"]
        args.init_positions = arg["init_positions"]
        args.init_agent_angles = arg["init_agent_angles"]
        args.alpha = arg["alpha"]
        args.max_episode_length = arg["max_episode_length"]
        args.enable_renderer = True
        args.all_directions = True

        skip = True
        if args.flip_env_id:
            # complete env ids based on env topology
            topology_groups = [[4, 8, 17], [12, 13, 18], [14, 19], [15, 20]]
            for t_group in topology_groups:
                if args.env_id in t_group:
                    print(args.env_id)
                    args.env_id = random.choice(list(set(t_group) - set([args.env_id])))
                    print(args.env_id)
                    skip = False
        if args.flip_env_id and skip:
            continue

        # change strength & goal item by same delta
        if args.flip_st_sz:
            delta = random.choice([-1, 1])
            args.strengths = [max(min(st + delta, 3), 0) for st in args.strengths]
            args.sizes = [max(min(sz + delta, 3), 0) for sz in args.sizes]

        def flip_ud(lm_order):  # 0
            return lm_order[::-1]

        def flip_lr(lm_order):  # 1
            return lm_order[:2][::-1] + lm_order[2:][::-1]

        def flip_diag1(lm_order):  # 2
            return [lm_order[2], lm_order[1], lm_order[0], lm_order[3]]

        def flip_diag2(lm_order):  # 3
            return [lm_order[0], lm_order[3], lm_order[2], lm_order[1]]

        landmark_color_order = [0, 1, 2, 3]
        if args.flip_env is not None:
            args.flip_env = np.random.choice(list(range(4)))
            if args.flip_env == 0:
                landmark_color_order = flip_ud(landmark_color_order)
            elif args.flip_env == 1:
                landmark_color_order = flip_lr(landmark_color_order)
            elif args.flip_env == 2:
                landmark_color_order = flip_diag1(landmark_color_order)
            elif args.flip_env == 3:
                landmark_color_order = flip_diag2(landmark_color_order)

        agent_order = [1, 2]
        item_order = [3, 4]
        if args.random_colors_agents:
            random.shuffle(agent_order)
            random.shuffle(item_order)
        # agent_order = [2,1]
        agent = MCTS_particle_2agents(args)
        start = timeit.default_timer()
        # print('path',path)
        if args.flip_env_id or args.flip_st_sz:
            # rerun planner
            agent.plan(nb_episodes=args.max_nb_episodes, record=True)
        else:
            agent.replay(
                arg,
                path,
                episode_id=episode_id,
                agent_order=agent_order,
                item_order=item_order,
                landmark_color_order=landmark_color_order,
                flip_env=args.flip_env,
            )
            # rotate_env=args.rotate_env)
        end = timeit.default_timer()
        print(end - start)
