# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import timeit
import math

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
    default=5,
    metavar="LENGTH",
    help="Maximum episode length",
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
parser.add_argument("--planning-agent", type=int, default=0, help="Planning agent")
parser.add_argument(
    "--mind-dim", type=int, default=128, help="Dimension of mind representation"
)
parser.add_argument("--action_freq", type=int, default=1, help="Action frequency")
parser.add_argument("--verbose", type=int, default=1, help="How much info to display")
parser.add_argument("--env-id", type=int, default=13, help="Environment id")
parser.add_argument(
    "--goals", nargs="*", default=[None, None], help="Goals of the agents"
)  # [41,35],[11,55], [62,26]protecting, [11,26] put in room/touch
parser.add_argument(
    "--goal1", nargs="*", default=["LMO", 0, 0, 1], help="Goals of the agents"
)
parser.add_argument(
    "--goal2", nargs="*", default=["LMO", 0, 0, 1], help="Goals of the agents"
)
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
    "--levels", nargs="*", type=int, default=[1, 1], help="Levels of the agents"
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
    help="True - save with date; False - only planning params",
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
    "--human-simulation",
    action="store_true",
    default=False,
    help="True - human-controlled; False - planner",
)


if __name__ == "__main__":
    args = parser.parse_args()

    ###################################
    # #parse goals for bash
    if len(args.goal1) > 0:
        goals1, goals2 = [], []
        for idx in range(0, len(args.goal1), 4):
            if args.goal1[idx] != "stop":
                goals1.append(
                    [
                        str(args.goal1[idx]),
                        int(args.goal1[idx + 1]),
                        int(args.goal1[idx + 2]),
                        int(args.goal1[idx + 3]),
                    ]
                )
            else:
                goals1.append(["stop"])
        for idx in range(0, len(args.goal2), 4):
            if args.goal2[idx] != "stop":
                goals2.append(
                    [
                        str(args.goal2[idx]),
                        int(args.goal2[idx + 1]),
                        int(args.goal2[idx + 2]),
                        int(args.goal2[idx + 3]),
                    ]
                )
            else:
                goals2.append(["stop"])
        args.goals = goals1 + goals2
        print("args.goals", args.goals)
    ###################################

    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + k + ": " + str(v))

    agent = MCTS_particle_2agents(args)
    start = timeit.default_timer()
    agent.plan(nb_episodes=args.max_nb_episodes, record=True)
    end = timeit.default_timer()
    print(end - start)
