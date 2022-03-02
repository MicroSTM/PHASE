from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from pathlib import Path
import sys
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing
from statistics import mode
import gc

from MCTS import *
from utils import *
from envs.box2d import *

from MCTS.MCTS_particles import MCTS_Particle

np.set_printoptions(precision=2, suppress=True)


def _str2class(str):
    return getattr(sys.modules[__name__], str)


def select_action(
    agent_id,
    goals,
    belief_particles,
    certainty,
    FOVs,
    attached,
    nb_steps,
    sim_envs,
    mcts,
    expected_action,
):
    if len(belief_particles[0]) == sim_envs[agent_id].num_agents:
        sim_envs[agent_id].set_state(
            belief_particles[0][: sim_envs[agent_id].num_agents],
            belief_particles[0][sim_envs[agent_id].num_agents :],
            nb_steps,
        )
    else:
        sim_envs[agent_id].set_state(
            belief_particles[0][: sim_envs[agent_id].num_agents],
            belief_particles[0][sim_envs[agent_id].num_agents :],
            certainty[0],
            FOVs,
            nb_steps,
        )

    initState = belief_particles  # sim_envs[agent_id].get_state(agent_id)
    # print(initState[agent_id])
    # print('initState', agent_id, [s['pos'] for s in initState])
    if sim_envs[agent_id].is_terminal(initState, nb_steps):
        terminal = True
        res[sample_id] = None
        return
    rootAction = None  # self.sim_envs[agent_id].action_space[-1] #random.choice(self.env.action_space)
    rootNode = Node(
        id=0,
        action=None,
        state_set=initState,
        certainty_set=certainty,
        num_visited=0,
        sum_value=0,
        is_expanded=True,
    )
    currNode = rootNode
    next_root, action, children_visit, action_idx = mcts[agent_id].run(
        currNode, nb_steps, expected_action
    )
    # expected_vels[agent_id] = self.sim_envs[agent_id].agents_vel[agent_id]
    # print(rootNode)

    return action, action_idx, np.array(children_visit)


class MCTS_particle_2agents:
    """
    MCTS for 2 agents
    """

    def __init__(self, args):
        # random seed
        random.seed(args.seed)
        # specify environment
        maze_sampler = MazeSampler()
        maze_sampler.gen_env_defs()
        self.maze_sampler = maze_sampler
        self.main_env = _str2class(args.exp_name)(
            action_type=args.action_type,
            maze_sampler=maze_sampler,
            goals=args.goals,
            strengths=args.strengths,
            sizes=args.sizes,
            densities=args.densities,
            init_positions=args.init_positions,
            action_space_types=args.action_space_types,
            costs=args.costs,
            temporal_decay=args.temporal_decay,
            visibility=args.visibility,
            num_agents=args.num_agents,
            num_items=args.num_items,
            all_directions=args.all_directions,
            enable_renderer=args.enable_renderer,
            render_separate_particles=args.save_particles_separately,
            random_colors=args.random_colors,
            random_colors_agents=args.random_colors_agents,
            full_obs=[f > 0 for f in args.full_obs],
            n_particles=args.n_particles,
            belief_prior=args.belief_prior,
            init_agent_angles=args.init_agent_angles,
            use_all_init_grid_pos=args.use_all_init_grid_pos,
            human_simulation=args.human_simulation,
        )
        self.env = _str2class(args.exp_name)(
            action_type=args.action_type,
            maze_sampler=maze_sampler,
            goals=args.goals,
            strengths=args.strengths,
            sizes=args.sizes,
            densities=args.densities,
            init_positions=args.init_positions,
            action_space_types=args.action_space_types,
            costs=args.costs,
            temporal_decay=args.temporal_decay,
            visibility=args.visibility,
            num_agents=args.num_agents,
            num_items=args.num_items,
            all_directions=args.all_directions,
            enable_renderer=False,  # args.enable_renderer,
            random_colors=args.random_colors,
            random_colors_agents=args.random_colors_agents,
            full_obs=[f > 0 for f in args.full_obs],
            n_particles=args.n_particles,
            belief_prior=args.belief_prior,
            init_agent_angles=args.init_agent_angles,
            use_all_init_grid_pos=args.use_all_init_grid_pos,
        )
        # self.env.particle_history = [[] for _ in range(self.env.num_agents)]
        self.num_agents = self.env.num_agents
        self.sim_envs = [
            _str2class(args.exp_name)(
                action_type=args.action_type,
                maze_sampler=maze_sampler,
                goals=args.goals,
                strengths=args.strengths,
                sizes=args.sizes,
                densities=args.densities,
                init_positions=args.init_positions,
                action_space_types=args.action_space_types,
                costs=args.costs,
                visibility=args.visibility,
                temporal_decay=args.temporal_decay,
                num_agents=args.num_agents,
                num_items=args.num_items,
                all_directions=args.all_directions,
                enable_renderer=False,  # args.enable_renderer,
                random_colors=args.random_colors,
                random_colors_agents=args.random_colors_agents,
                full_obs=[f > 0 for f in args.full_obs],
                n_particles=args.n_particles,
                belief_prior=args.belief_prior,
                init_agent_angles=args.init_agent_angles,
                use_all_init_grid_pos=args.use_all_init_grid_pos,
            )
            for _ in range(self.num_agents)
        ]
        for agent_id in range(self.num_agents):
            self.sim_envs[agent_id].action_idxs = [[] for _ in range(self.num_agents)]
        self.simulation = {}
        self.action_size = self.env.action_size
        self.planning_agent = args.planning_agent
        self.max_episode_length = args.max_episode_length
        self.num_processes = args.nb_processes
        self.num_samples = args.nb_samples
        self.max_rollout_steps = args.max_rollout_steps
        self.nb_simulations = args.nb_simulations
        self.cInit = args.cInit
        self.cBase = args.cBase
        self.env_id = args.env_id
        self.levels = args.levels
        self.goals = args.goals
        self.alpha = args.alpha
        self.strengths = args.strengths
        self.sizes = args.sizes
        self.densities = args.densities
        self.init_positions = args.init_positions
        self.action_space_types = args.action_space_types
        self.costs = args.costs
        self.temporal_decay = (args.temporal_decay,)
        self.init_times = args.init_times
        self.init_goals = args.init_goals
        self.action_type = args.action_type
        self.plan_length = args.plan_length
        self.execute_length = args.execute_length
        self.all_directions = args.all_directions
        self.simulate_n_steps = args.simulate_n_steps
        self.simulate_score_thres = args.simulate_score_thres
        self.beta = args.beta
        self.full_obs = args.full_obs
        self.enable_renderer = args.enable_renderer
        self.enable_belief_renderer = args.enable_belief_renderer
        self.angles = args.init_agent_angles
        self.args = args
        if args.save_date:
            self.record_dir = (
                args.record_dir
                + "/"
                + args.exp_name
                + "/D{}_{}_{}{}_E{}_G{}_ST{}_SZ{}_P{}_A{}_C{}_D{}_M{}_AN{}_MCTS_L{}".format(
                    datetime.now().strftime("%m%d%y_%H%M%S"),
                    "_".join(["{}".format(f) for f in self.full_obs]),
                    "F" if self.action_type == "force" else "I",
                    8 if self.all_directions else 1,
                    self.env_id,
                    "_".join(["{}".format(g) for g in self.goals]),
                    "_".join(["{}".format(s) for s in self.strengths]),
                    "_".join(["{}".format(s) for s in self.sizes]),
                    "_".join(["{}".format(p) for p in self.init_positions]),
                    "_".join(["{}".format(a) for a in self.action_space_types]),
                    "_".join(["{}".format(c) for c in self.costs]),
                    "_".join(["{}".format(d) for d in self.temporal_decay]),
                    "_".join(["{}".format(d) for d in self.alpha]),
                    "_".join(["{}".format(d) for d in self.angles]),
                    self.max_episode_length,
                )
            )
        else:
            self.record_dir = (
                args.record_dir
                + "/"
                + args.exp_name
                + "/{}_{}{}_E{}_G{}_ST{}_SZ{}_P{}_A{}_C{}_AN{}_MCTS_L{}".format(
                    "_".join(["{}".format(f) for f in self.full_obs]),
                    "F" if self.action_type == "force" else "I",
                    8 if self.all_directions else 1,
                    self.env_id,
                    "_".join(["{}".format(g) for g in self.goals]),
                    # self.alpha[0], self.alpha[1],
                    "_".join(["{}".format(s) for s in self.strengths]),
                    "_".join(["{}".format(s) for s in self.sizes]),
                    "_".join(["{}".format(p) for p in self.init_positions]),
                    "_".join(["{}".format(a) for a in self.action_space_types]),
                    "_".join(["{}".format(c) for c in self.costs]),
                    "_".join(["{}".format(d) for d in self.angles]),
                    self.max_episode_length,
                )
            )
        p = Path(self.record_dir)
        if not p.is_dir():
            p.mkdir(parents=True)
        self.video_dir = str(p)

    def informing_goal_decomposition(self, env, agent_id):
        """env+agent_id of observing agent"""
        # TODO: - add 'friendly' mode as condition?

        # TODO find thres, should compare as lower bound for plan change.
        plan_score = self.simulation["plan_score"][-1]
        if (
            plan_score <= self.simulate_score_thres
            or len(self.simulation["sim_goals"]) > 0
            and self.last_goals[1 - agent_id] == self.simulation["sim_goals"][-1][-1]
        ):
            # TODO change to grabbing & support grabbing?

            # stop --> change to observe
            return env.get_subgoals_avoid(agent_id)
        else:
            return env.get_subgoals_goto(agent_id, ["TE", agent_id, 1 - agent_id, +1])

    def hiding_goal_decomposition(self, env, agent_id, goal=["V", 1, 0, +1]):
        if env.env_id == 0:
            print("can't hide")
            return goal  # can't hide

        goal_agent = 1 - agent_id  # chasing agent
        entity_pos = env.agent_states[goal_agent]["pos"]  # chasing agent pos
        entity_room = env._get_room_id(entity_pos)  # chasing agent room
        agent_room = env._get_room_id(
            env.agent_states[agent_id]["pos"]
        )  # hiding agent room

        if (
            entity_room != agent_room
            and env._is_blocked(
                env.doors[(entity_room, agent_room)], SIZE[env.sizes[agent_id]]
            )
            and not env.in_field_of_view(goal_agent, ("agent", agent_id))
        ):
            print("can't hide")
            return ["stop"]  # do nothing, chaser is blocked

        # hide by simulation blind spots, furthest from chasing agent
        entity_sim_fov = self.simulation["accum_fovs"][-1].flatten()
        for step in range(self.simulate_n_steps + 1):
            hiding_opts = np.array(env.all_init_pos)[entity_sim_fov.flatten() == step]
            hiding_dist = [
                env._get_dist_pos(entity_pos, hiding_opt) for hiding_opt in hiding_opts
            ]
            # print(hiding_opts)
            # print(hiding_dist)
            if len(hiding_opts) > 0:
                # hiding_pos = random.choice(hiding_opts)
                hiding_pos = hiding_opts[np.argmax(hiding_dist)]
                print("hiding_pos", hiding_pos)
                room_id = env._get_room_id(hiding_pos)
                if env._get_room_id(env.agent_states[agent_id]["pos"]) == room_id:
                    return ["PA", agent_id, hiding_pos, +1]
                else:
                    return env.get_subgoals_goto(
                        agent_id, ["RA", agent_id, room_id, +1]
                    )  # goto room first

        # hide in the furthest room chasing can't see
        room_goals = []
        rooms_dist = []  # from chasing
        room_pos = [
            [
                env.room.position[0] - env.room_dim[0] // 4,
                env.room.position[1] + env.room_dim[1] // 4,
            ],
            [
                env.room.position[0] + env.room_dim[0] // 4,
                env.room.position[1] + env.room_dim[1] // 4,
            ],
            [
                env.room.position[0] + env.room_dim[0] // 4,
                env.room.position[1] - env.room_dim[1] // 4,
            ],
            [
                env.room.position[0] - env.room_dim[0] // 4,
                env.room.position[1] - env.room_dim[1] // 4,
            ],
        ]
        for room_id in range(4):
            if entity_room != room_id and agent_room != room_id:
                if not env.room_in_fov(goal_agent, room_id) and not env._is_blocked(
                    env.doors[(agent_room, room_id)], SIZE[env.sizes[agent_id]]
                ):
                    print("room_id", room_id)
                    room_goals.append(
                        env.get_subgoals_goto(agent_id, ["RA", agent_id, room_id, +1])
                    )  # hide in a room
                    # rooms_dist.append(env._get_dist_room(entity_pos, room_id))
                    rooms_dist.append(env._get_dist_pos(entity_pos, room_pos[room_id]))
        print("hiding room", list(zip(room_goals, rooms_dist)))
        if len(room_goals) > 0:
            return room_goals[np.argmax(rooms_dist)]

        print("hiding by running")
        return [
            "TE",
            goal_agent,
            agent_id,
            -1,
        ]  # agent 0 not touching agent 1 (chasing not touching hiding)

    def goal_decomposition(self, env, agent_id, goal, mode="neutral"):
        # print('type',goal,type(goal),agent_id)
        # TODO multiple goals check
        if len(goal) == 2:
            return env.multiple_goals(goal, agent_id)
        """TODO: more general decomposition"""
        if mode == "neutral":
            if goal[0] == "LMO" and goal[3] == 1:
                return env.get_subgoals_put(agent_id, goal)
            elif (goal[0] == "LMA" and goal[3] == 1) or (
                goal[0] == "TE" and goal[3] == 1
            ):
                return env.get_subgoals_goto(agent_id, goal)
        elif mode == "friendly":
            if goal[0] == "LMO" and goal[3] == 1:
                return env.get_subgoals_put_help(agent_id, goal)
            elif (goal[0] == "LMA" and goal[3] == 1) or (
                goal[0] == "TE" and goal[3] == 1
            ):
                return env.get_subgoals_goto_help(agent_id, goal)
        else:  # use other agent's goal in adversarial mode
            if goal[0] == "LMO" and goal[3] == 1:
                return env.get_subgoals_put_hinder(agent_id, goal)
            elif (goal[0] == "LMA" and goal[3] == 1) or (
                goal[0] == "TE" and goal[3] == 1
            ):
                return env.get_subgoals_goto_hinder(agent_id, goal)

        return goal

    def goal_decomposition_particles(
        self,
        belief_states,
        certainty,
        FOVs,
        steps,
        env,
        agent_id,
        goal,
        mode="neutral",
        undesired_subgoals=None,
    ):
        print("goal_decomposition_particles", goal)
        goals = []
        costs = {}
        nb_particles = len(belief_states)
        for belief_state in belief_states:
            env.set_state(
                belief_state[: self.num_agents],
                belief_state[self.num_agents :],
                certainty,
                FOVs,
                steps,
            )
            env.update_field_of_view()
            goals.append(self.goal_decomposition(env, agent_id, goal, mode))
            cost = env.get_cost(agent_id, goal) * 0.05
            if undesired_subgoals is not None and goals[-1] in undesired_subgoals:
                cost += 10
            if tuple(goals[-1]) in [tuple(c) for c in costs]:
                costs[tuple(goals[-1])].append(cost)
            else:
                costs[tuple(goals[-1])] = [cost]
        # print('subgoals for all particles:', goals)
        # print('undesired_subgoals:', undesired_subgoals)
        Qs = {
            goal: len(cost_list) / nb_particles - np.mean(cost_list)
            for goal, cost_list in costs.items()
        }
        print("Qs:", Qs)
        # TODO item_id
        selected_goal = None
        if len(goal) == 2:  # TODO: more general; currently specifically for stealing
            for goal_particle in goals:
                if goal_particle[-1] == -1:
                    selected_goal = goal_particle
                    break
            if selected_goal is None and ["TE", agent_id, 1 - agent_id, +1] in goals:
                selected_goal = ["TE", agent_id, 1 - agent_id, +1]
        elif ["TE", agent_id, 2, +1] in goals:
            selected_goal = ["TE", agent_id, 2, +1]
        elif ["TE", agent_id, 3, +1] in goals:
            selected_goal = ["TE", agent_id, 3, +1]
        elif ["TE", agent_id, 1 - agent_id, +1] in goals:
            selected_goal = ["TE", agent_id, 1 - agent_id, +1]
        elif self.last_goals[agent_id] in goals:
            selected_goal = self.last_goals[agent_id]
        if selected_goal is None:
            # selected_goal = max(set(goals), key=goals.count)
            selected_goal = max(Qs, key=Qs.get)
        selected_goal = list(selected_goal)
        selected_belief_states = [
            belief_state
            for belief_state, g in zip(belief_states, goals)
            if g == selected_goal
        ]
        # if selected_goal == 25 + agent_id:
        #     for belief_state in selected_belief_states:
        #         pos = belief_state[self.num_agents]['pos']
        #         cell = env.world_point_to_grid_cell(pos[0], pos[1])
        #         print(pos, FOVs[agent_id][cell[0]][cell[1]])
        return selected_goal, selected_belief_states

    def all_goal_decomposition_particles(
        self, belief_states, certainty, FOVs, steps, env, agent_id, goal, mode="neutral"
    ):
        goals = []
        for belief_state in belief_states:
            env.set_state(
                belief_state[: self.num_agents],
                belief_state[self.num_agents :],
                certainty,
                FOVs,
                steps,
            )
            goals.append(self.goal_decomposition(env, agent_id, goal, mode))
        return goals

    def rollout(self, record=False, episode_id=None):
        nb_steps = 0
        if record:
            self.main_env.setup(
                self.env_id,
                self.planning_agent,
                self.max_episode_length,
                record_path=self.video_dir
                + "/R{}_{}_PL{}_EL{}_{}_{}_s{}_r{}_cI{}_cB{}_e{}.mp4".format(
                    self.alpha[0],
                    self.alpha[1],
                    self.plan_length,
                    self.execute_length,
                    self.levels[0],
                    self.levels[1],
                    self.nb_simulations,
                    self.max_rollout_steps,
                    self.cInit,
                    self.cBase,
                    self.episode_id,
                ),
                contact_listen=False,
            )
        else:
            self.main_env.setup(
                self.env_id, self.planning_agent, self.max_episode_length
            )
        # self.env.start(initial_push=(self.planning_agent == 1))

        self.main_env.start(self.video_dir)
        if self.main_env.use_all_init_grid_pos:
            return
        final_modes = [None] * self.num_agents
        final_subgoals = [None] * self.num_agents
        all_modes = []
        rewards = []
        self.subgoal_history = {agent_id: [] for agent_id in range(self.num_agents)}
        while self.main_env.running:
            (
                modes,
                plans,
                Vs,
                subgoals,
                cur_all_modes,
                sampled_beliefs,
            ) = self.plan_select_high_level_goals()
            for agent_id, subgoal_list in enumerate(subgoals):
                self.subgoal_history[agent_id] += list(subgoal_list)
            print("plans", plans)
            # print('sampled_beliefs',sampled_beliefs)
            all_modes.append(cur_all_modes)
            T = min(min(len(plans[0]), len(plans[1])), self.execute_length)
            for t in range(T):
                if not self.main_env.running:
                    break
                for agent_id in range(self.num_agents):
                    self.main_env.send_action(agent_id, plans[agent_id][t])
                    if final_modes[agent_id] is None:
                        final_modes[agent_id] = [modes[agent_id]]
                    else:
                        final_modes[agent_id] += [modes[agent_id]]
                    if final_subgoals[agent_id] is None:
                        final_subgoals[agent_id] = [subgoals[agent_id][t]]
                    else:
                        final_subgoals[agent_id] += [subgoals[agent_id][t]]
                if self.enable_belief_renderer:
                    self.main_env.render_FOV(
                        record_dir=self.video_dir, path_extension="_" + str(episode_id)
                    )
                    # print('main state',self.main_env.get_state())
                    for agent_id in range(self.num_agents):
                        self.main_env._display_imagined(
                            agent_id,
                            sampled_beliefs[agent_id][t],
                            record_dir=self.video_dir,
                            path_extension="_" + str(episode_id),
                        )
                        # print('belief state',sampled_beliefs[agent_id][t])
                self.main_env.step()
                reward = [
                    self.main_env.get_reward_state(
                        agent_id, self.main_env.get_state(), action=plans[agent_id][t]
                    )
                    for agent_id in range(self.num_agents)
                ]
                rewards.append(reward)
                if self.main_env.steps >= self.max_episode_length:
                    break
            if self.main_env.steps >= self.max_episode_length:
                break

        if record:
            path = (
                self.video_dir
                + "/R{}_{}_PL{}_EL{}_{}_{}_s{}_r{}_cI{}_cB{}_e{}.pik".format(
                    self.alpha[0],
                    self.alpha[1],
                    self.plan_length,
                    self.execute_length,
                    self.levels[0],
                    self.levels[1],
                    self.nb_simulations,
                    self.max_rollout_steps,
                    self.cInit,
                    self.cBase,
                    self.episode_id,
                )
            )
            pickle.dump(
                {
                    "trajectories": self.main_env.trajectories,
                    "actions": self.main_env.actions,
                    "wall_segs": self.main_env.wall_segs,
                    "landmark_centers": self.main_env.landmark_centers,
                    "sizes": self.main_env.sizes_value,
                    "strengths": self.main_env.strengths_value,
                    "entity_color_code": self.main_env.entity_color_code,
                    "goals": self.goals,
                    "modes": final_modes,
                    "subgoals": final_subgoals,
                    "rewards": rewards,
                    "all_modes": all_modes,
                    "simulations": self.simulation,
                    "sampled_beliefs": sampled_beliefs,
                },
                open(path, "wb"),
            )

    def plan_select_high_level_goals(self):

        """TODO: check if need to help"""
        plans = {}
        subgoals = {}
        trajs = {}
        Vs = {}
        sampled_beliefs = {}

        # if self.alpha[0] > 1e-6:
        #     modes_all = [('neutral', 'neutral'), ('friendly', 'neutral')]
        # elif self.alpha[1] > 1e-6:
        #     modes_all = [('neutral', 'neutral'), ('neutral', 'friendly')]
        # elif self.alpha[0] < -1e-6:
        #     modes_all = [('neutral', 'neutral'), ('adversarial', 'neutral')]
        # elif self.alpha[1] < -1e-6:
        #     modes_all = [('neutral', 'neutral'), ('neutral', 'adversarial')]
        # else:
        #     modes_all = [('neutral', 'neutral')]
        if self.alpha[0] > 1e-6:
            modes_all = [("friendly", "neutral")]
        elif self.alpha[0] < -1e-6:
            modes_all = [("adversarial", "neutral")]
        else:
            modes_all = [("neutral", "neutral")]

        print("modes_all", modes_all)
        manager = multiprocessing.Manager()
        res_plans = manager.dict()
        res_subgoals = manager.dict()
        res_trajs = manager.dict()
        res_Vs = manager.dict()
        res_sampled_beliefs = manager.dict()

        jobs = []
        if len(modes_all) > 1:
            """TODO: inherent previous particles"""
            for process_id, modes in enumerate(modes_all):
                p = multiprocessing.Process(
                    target=self.plan_given_high_level_goals,
                    args=(
                        modes,
                        self.main_env,
                        self.env,
                        self.sim_envs,
                        res_plans,
                        res_subgoals,
                        res_trajs,
                        res_Vs,
                        res_sampled_beliefs,
                        True,
                    ),
                )
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        else:
            self.plan_given_high_level_goals(
                modes_all[0],
                self.main_env,
                self.env,
                self.sim_envs,
                res_plans,
                res_subgoals,
                res_trajs,
                res_Vs,
                res_sampled_beliefs,
                False,
            )

        print("modes_all", modes_all)
        for modes in modes_all:
            plans[modes] = res_plans[modes]
            subgoals[modes] = res_subgoals[modes]
            trajs[modes] = res_trajs[modes]
            Vs[modes] = res_Vs[modes]
            sampled_beliefs[modes] = res_sampled_beliefs[modes]

        print(Vs)

        best_V = -1e6
        best_plans, best_subgoals, best_trajs, best_plan_beliefs = (
            None,
            None,
            None,
            None,
        )
        best_modes = ("neutral", "neutral")
        for mode_1 in ["adversarial", "neutral", "friendly"]:
            for mode_2 in ["adversarial", "neutral", "friendly"]:
                modes = (mode_1, mode_2)
                if modes in Vs:
                    V_sum = Vs[modes][0]
                    if V_sum > best_V:
                        best_V = V_sum
                        best_plans = plans[modes]
                        best_subgoals = subgoals[modes]
                        best_trajs = trajs[modes]
                        best_plan_beliefs = sampled_beliefs[modes]
                        best_modes = (mode_1, mode_2)

        print("high level goals:", best_modes, best_V)

        # input('press any key to continue...')

        return (
            best_modes,
            best_plans,
            best_V,
            best_subgoals,
            {"trajs": trajs, "plans": plans, "subgoals": subgoals, "Vs": Vs},
            best_plan_beliefs,
        )

    def plan_given_high_level_goals(
        self,
        modes,
        main_env,
        env,
        sim_envs,
        res_plans,
        res_subgoals,
        res_trajs,
        res_Vs,
        res_sampled_beliefs,
        restart=False,
    ):
        plan_length = self.plan_length
        subgoal_history = copy.deepcopy(self.subgoal_history)
        undesired_subgoals = {agent_id: None for agent_id in range(self.num_agents)}

        if restart or main_env.steps == 0:
            nb_steps = 0
            """TODO: remove planning agent from global environment?"""
            env.setup(
                self.env_id,
                self.planning_agent,
                self.max_episode_length,
                contact_listen=False,
            )  # , record_path=self.video_dir + \
            # '/{}_{}_s{}_r{}_cI{}_cB{}_e{}_{}_{}_{}{}.mp4'.format(
            #                                         self.levels[0], self.levels[1],
            #                                         self.nb_simulations, self.max_rollout_steps,
            #                                         self.cInit, self.cBase, self.episode_id,
            #                                         self.main_env.steps+1, self.main_env.steps+1+self.plan_length,
            #                                         modes[0][0], modes[1][0]))
            env.start()
            # if self.main_env.num_items:
            #     self.env.set_state(self.main_env.agents_pos, self.main_env.agents_vel, self.main_env.items_pos, self.main_env.items_vel, self.main_env.attached, self.main_env.steps)
            # else:
            #     self.env.set_state(self.main_env.agents_pos, self.main_env.agents_vel, self.main_env.steps)

            for t in range(self.main_env.steps):
                for agent_id in range(self.num_agents):
                    print("main env actions", self.main_env.actions[agent_id][t])
                    env.send_action(agent_id, self.main_env.actions[agent_id][t])
                env.step()
                env.belief_step(
                    [
                        self.main_env.actions[agent_id][t]
                        for agent_id in range(self.num_agents)
                    ]
                )
        else:
            nb_steps = main_env.steps
        env.reset_history()

        curr_belief = [
            env.agents[agent_id].beliefs.particles
            for agent_id in range(self.num_agents)
        ]
        # print('curr_belief1',curr_belief)

        for agent_id in range(self.num_agents):
            sim_envs[agent_id].setup(self.env_id, agent_id, self.max_episode_length)
            sim_envs[agent_id].start()
            # for t in range(self.env.steps):
            #     for agent_id_tmp in range(self.num_agents):
            #         self.sim_envs[agent_id].send_action(self.env.actions[self.num_agents])
            if agent_id == 2:
                sim_envs[agent_id].set_state(
                    env.agent_states,
                    env.item_states,
                    [agent.certainty.copy() for agent in env.agents],
                    [agent.field_of_view.copy() for agent in env.agents],
                    main_env.steps,
                )
            else:
                if env.num_items:
                    # print('belief for plan:', curr_belief[agent_id][agent_id]['attached'], env.attached)
                    # TODO change choice of particle 0
                    sim_envs[agent_id].set_state(
                        curr_belief[agent_id][0][: self.num_agents],
                        curr_belief[agent_id][0][self.num_agents :],
                        [agent.certainty.copy() for agent in env.agents],
                        [agent.field_of_view.copy() for agent in env.agents],
                        main_env.steps,
                    )
                    # sim_envs[agent_id].set_state(env.agent_states, env.item_states, env.attached, main_env.steps)
                else:
                    sim_envs[agent_id].set_state(
                        env.agents_pos, env.agents_vel, main_env.steps
                    )

        # print(nb_steps, env.agents_pos, env.agents_vel)

        # gt_actions = ['down'] * 8 + ['downleft'] * 6 + ['left', 'downleft', 'noforce', 'upleft', 'downleft', 'upleft', 'stop', 'stop', 'left', 'down', 'noforce', 'stop', 'down', 'downleft', 'down', 'downleft']
        # gt_actions = ['down'] * 8 + ['downleft'] * 6 + ['left', 'downleft', 'noforce', 'upleft', 'downleft', 'upleft', 'stop', 'stop', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'down']
        goals = [None] * self.num_agents
        Vs = [0] * self.num_agents
        sampled_beliefs = [None] * self.num_agents
        certainty_history = []
        FOV_history = []
        self.simulation["sim_actions"] = []
        self.simulation["action_probs"] = []
        self.simulation["action_idx"] = []
        self.simulation["world_states"] = []
        self.simulation["sim_goals"] = []
        self.simulation["plan_score"] = [0]  # 0.5
        self.simulation["accum_fovs"] = []

        for agent_id, belief in enumerate(curr_belief):
            if sampled_beliefs[agent_id] is None:
                sampled_beliefs[agent_id] = [belief]
            else:
                sampled_beliefs[agent_id].append(belief)
        self.last_goals = {agent_id: None for agent_id in range(self.num_agents)}

        while env.running:
            certainty = [agent.certainty.copy() for agent in env.agents]
            FOVs = [agent.field_of_view.copy() for agent in env.agents]
            selected_belief_states = {}

            # hiding, plan ahead, before env
            if (
                env.steps % self.simulate_n_steps == 0
                and self.goals[agent_id][0] == "V"
            ):
                self.hiding_sim(certainty, FOVs, sim_envs, env, main_env)

            for agent_id, goal in enumerate(self.goals):
                if modes[agent_id] != "neutral":
                    print("!=neutral", sim_envs[agent_id].get_state())
                    (
                        decomposed_goal,
                        selected_belief_states[agent_id],
                    ) = self.goal_decomposition_particles(
                        curr_belief[agent_id],
                        certainty,
                        FOVs,
                        main_env.steps,
                        sim_envs[agent_id],
                        agent_id,
                        self.goals[1 - agent_id],
                        mode=modes[agent_id],
                    )
                else:
                    if goal[0] == "OBS":  # informing
                        decomposed_goal = self.informing_goal_decomposition(
                            sim_envs[agent_id], agent_id
                        )
                        selected_belief_states[agent_id] = curr_belief[agent_id].copy()

                        if (
                            goals[1 - agent_id] is not None
                            and env.goals[agent_id][0] == "TE"
                            and decomposed_goal == ["stop"]
                        ):
                            # TODO check what happens if previously had correct subgoal
                            undesired_subgoals[1 - agent_id] = subgoal_history[
                                1 - agent_id
                            ]
                            print(
                                "undesired_subgoals:", undesired_subgoals[1 - agent_id]
                            )
                            # add costs to wrong subgoals (only consider going to rooms subgoals)

                            # #resmaple all particles that lead to same subgoal in 1-agent_id
                            # other_decomposed_goals = self.all_goal_decomposition_particles(curr_belief[1-agent_id], certainty, FOVs, main_env.steps, sim_envs[1-agent_id], 1-agent_id, self.goals[1-agent_id])
                            # print(other_decomposed_goals)

                            # # n_sim_last_goals = goals[1-agent_id][env.steps-2*self.simulate_n_steps:env.steps-self.simulate_n_steps] #when the decision the inform was made
                            # # goal2change = max(set(n_sim_last_goals), key=n_sim_last_goals.count) #mode(n_sim_last_goals) #mode cannot handle non-unique most freq
                            # # print('goal2change ', goal2change, ' last subgoals ', n_sim_last_goals)

                            # last_sim_goals = self.simulation['sim_goals'][-1]
                            # goal2keep = max(set(last_sim_goals), key=last_sim_goals.count)
                            # print('goal2keep ', goal2keep, ' last sim subgoals ', last_sim_goals)

                            # #repeated reampling
                            # for _ in range(5):
                            #     # #change based on actual last steps
                            #     # idxs2change = np.where(np.array(other_decomposed_goals)==goal2change)[0]
                            #     # print('resampling subgoal ',goal2change,' idxs ', idxs2change)

                            #     #change based on sim subgoal decompos
                            #     idxs2change = np.where(np.array(other_decomposed_goals)!=goal2keep)[0]
                            #     print('resampling subgoal ',goal2keep,' idxs ', idxs2change)

                            #     env.resample_specified_particles(1-agent_id, idxs2change, self.goals[1-agent_id])
                            #     other_decomposed_goals = self.all_goal_decomposition_particles(env.agents[1-agent_id].beliefs.particles, certainty, FOVs, main_env.steps, sim_envs[1-agent_id], 1-agent_id, self.goals[1-agent_id])

                            # # #particle drop
                            # # idxs2keep = np.where(np.array(other_decomposed_goals)!=goal2change)[0]
                            # # env.agents[1-agent_id].beliefs.update([env.agents[1-agent_id].beliefs.particles[j] for j in idxs2keep])

                            # curr_belief[1-agent_id] = env.agents[1-agent_id].beliefs.particles
                    elif goal[0] == "V":  # hiding
                        decomposed_goal = self.hiding_goal_decomposition(
                            sim_envs[agent_id], agent_id
                        )
                        selected_belief_states[agent_id] = curr_belief[agent_id].copy()
                    else:
                        (
                            decomposed_goal,
                            selected_belief_states[agent_id],
                        ) = self.goal_decomposition_particles(
                            curr_belief[agent_id],
                            certainty,
                            FOVs,
                            main_env.steps,
                            sim_envs[agent_id],
                            agent_id,
                            goal,
                            undesired_subgoals=undesired_subgoals[agent_id],
                        )
                print(
                    "decomposed_goal",
                    decomposed_goal,
                    "#selected particles",
                    len(selected_belief_states[agent_id]),
                )
                env.goals[agent_id] = decomposed_goal
                subgoal_history[agent_id].append(decomposed_goal)
                self.last_goals[agent_id] = decomposed_goal
                print("decomposed goals", env.goals)
                for agent_id_tmp in range(self.num_agents):
                    sim_envs[agent_id_tmp].goals[agent_id] = decomposed_goal

            certainty_history.append(certainty)
            FOV_history.append(FOVs)

            for agent_id, goal in enumerate(self.env.goals):
                if goals[agent_id] is None:
                    goals[agent_id] = [goal]
                else:
                    goals[agent_id].append(goal)

            mcts = [[None] * self.num_agents for _ in range(2)]  # two levels
            for level in range(2):
                for agent_id in range(self.num_agents):
                    # specify MCTS
                    mcts[level][agent_id] = MCTS_Particle(
                        agent_id=agent_id,
                        action_space=sim_envs[agent_id].get_action_space(agent_id),
                        transition=sim_envs[
                            agent_id
                        ].transition,  # if level == 0 else self.transition,
                        # transition=self.transition,
                        # attached=sim_envs[agent_id].attached,
                        reward=sim_envs[agent_id].get_reward_state,
                        is_terminal=sim_envs[agent_id].is_terminal,
                        num_simulation=self.nb_simulations,
                        max_rollout_steps=self.max_rollout_steps,
                        c_init=self.cInit,
                        c_base=self.cBase,
                    )

            currState = env.get_state()

            """TODO: check global environment status update & terminal condition"""
            if env.is_terminal(currState, nb_steps):
                break
            terminal = False
            # expected_vels = list(env.agents_vel)
            actions = [[None] * self.num_agents, [None] * self.num_agents]
            # plans = [[None] * self.num_agents, [None] * self.num_agents]
            last_level_actions = [None] * self.num_agents
            for agent_id in range(self.num_agents):
                """TODO: remove this after generating examples"""
                if env.steps < self.init_times[agent_id]:
                    actions[0][agent_id] = "stop"
                    continue
                if env.goals[agent_id] == ["stop"]:
                    actions[0][agent_id] = "stop"
                    continue
                if env.goals[agent_id] == ["noforce"]:
                    actions[0][agent_id] = "noforce"
                    continue

                expected_action = last_level_actions[1 - agent_id]

                action, action_idx, children_visit = select_action(
                    agent_id,
                    self.goals,
                    selected_belief_states[agent_id],
                    [
                        [agent.certainty.copy() for agent in env.agents]
                        for _ in range(len(selected_belief_states[agent_id]))
                    ],
                    [agent.field_of_view.copy() for agent in env.agents],
                    env.attached,
                    env.steps,
                    sim_envs,
                    mcts[agent_id],
                    expected_action,
                )
                sim_envs[agent_id].action_idxs[agent_id].append(action_idx)

                if action is None:
                    terminal = True
                    break
                actions[0][agent_id] = action
                # plans[0][agent_id] = res[0]
                # print('level-1 agent:', agent_id, actions[0][agent_id])
                if terminal:
                    break
            if terminal:
                break

            # final_actions = [actions[self.levels[agent_id]][agent_id] for agent_id in range(self.num_agents)]
            final_actions = [
                actions[0][agent_id] for agent_id in range(self.num_agents)
            ]
            # final_plans = [plans[self.levels[agent_id]][agent_id] for agent_id in range(self.num_agents)]
            print("final actions:", final_actions)
            # if not self.env.is_far():
            for agent_id, action in enumerate(final_actions):
                env.send_action(agent_id, action)
            env.step()
            print("env state", env.get_state())
            env.belief_step(final_actions)
            curr_belief = [
                env.agents[agent_id].beliefs.particles
                for agent_id in range(self.num_agents)
            ]
            # print('curr_belief2',curr_belief)
            # print('confidence', [env.get_belief_certainty(agent_id) for agent_id in range(self.num_agents)])

            # observing agent simulation of other agent based on its belief (/gt) compared to actual env
            if (
                env.steps % self.simulate_n_steps == 0
                and any(self.args.full_obs)
                and self.goals[np.where(self.args.full_obs)[0][0]][0] == "OBS"
            ):
                self.informing_sim(
                    sampled_beliefs,
                    certainty_history,
                    FOV_history,
                    sim_envs,
                    env,
                    main_env,
                )

            rewards = [0] * self.num_agents
            individual_rewards = [0] * self.num_agents

            for agent_id in range(self.num_agents):
                individual_rewards[agent_id] = env.get_reward_state(
                    agent_id, env.get_state(), "stop", None, None, self.goals[agent_id]
                )
            print("individual_rewards:", individual_rewards)
            # for agent_id in range(self.num_agents):
            #     if modes[agent_id] == 'neutral':
            #         rewards[agent_id] =
            #     elif modes[agent_id] == 'friendly':
            #         rewards[agent_id] = env.get_reward_state(agent_id, env.get_state(agent_id), 'stop', None, None, self.goals[1 - agent_id])
            #     else:
            #         rewards[agent_id] = -env.get_reward_state(agent_id, env.get_state(agent_id), 'stop', None, None, self.goals[1 - agent_id])
            for agent_id in range(self.num_agents):
                rewards[agent_id] = (
                    individual_rewards[agent_id]
                    + self.alpha[agent_id] * individual_rewards[1 - agent_id]
                )
                Vs[agent_id] += rewards[agent_id]
                sampled_beliefs[agent_id] += [curr_belief[agent_id]]

            nb_steps += 1
            print(
                modes,
                nb_steps,
                "({})".format(env.steps),
                [agent_state["pos"] for agent_state in env.agent_states],
                [item_state["pos"] for item_state in env.item_states],
                env._get_room_id(env.agent_states[0]["pos"]),
                env._get_room_id(env.agent_states[1]["pos"]),
                actions,
                env.goals,
                [a is not None for a in env.attached],
                rewards,
            )
            if (
                env.steps >= self.max_episode_length
                or nb_steps - main_env.steps >= plan_length
            ):
                break

        res_plans[modes] = env.actions
        res_subgoals[modes] = goals
        res_trajs[modes] = env.trajectories
        res_Vs[modes] = Vs
        res_sampled_beliefs[modes] = sampled_beliefs
        # return self.env.trajectories, self.env.actions, goals, Vs

    def informing_sim(
        self, sampled_beliefs, certainty_history, FOV_history, sim_envs, env, main_env
    ):
        # observing agent should simulate other agent based on observing belief (/gt)
        agent_id = np.where(self.args.full_obs)[0][0]
        sim_goals = []
        sim_actions = []
        score = 0
        # start with belief 5 steps ago, then progress based on transition with recommended action
        curr_sim_state = sampled_beliefs[agent_id][
            -self.simulate_n_steps
        ]  # particles, all the same
        print("sim start pos", curr_sim_state[0][1 - agent_id]["pos"])
        certainty = certainty_history[-self.simulate_n_steps]  # for both agents
        FOVs = FOV_history[-self.simulate_n_steps]
        mcts = [[None] * self.num_agents for _ in range(2)]
        for sim_step in range(self.simulate_n_steps):
            # TODO main_env.steps
            sim_envs[1 - agent_id].set_state(
                curr_sim_state[0][: self.num_agents],
                curr_sim_state[0][self.num_agents :],
                certainty,
                FOVs,
                main_env.steps,
            )
            (
                sim_envs[1 - agent_id].goals[1 - agent_id],
                selected_belief_states,
            ) = self.goal_decomposition_particles(
                curr_sim_state,
                certainty,
                FOVs,
                main_env.steps,
                sim_envs[1 - agent_id],
                1 - agent_id,
                self.goals[1 - agent_id],
            )
            sim_goals.append(sim_envs[1 - agent_id].goals[1 - agent_id])
            mcts[agent_id][1 - agent_id] = MCTS_Particle(
                agent_id=1 - agent_id,
                action_space=sim_envs[1 - agent_id].get_action_space(1 - agent_id),
                transition=sim_envs[1 - agent_id].transition,
                reward=sim_envs[1 - agent_id].get_reward_state,
                is_terminal=sim_envs[1 - agent_id].is_terminal,
                num_simulation=self.nb_simulations,
                max_rollout_steps=self.max_rollout_steps,
                c_init=self.cInit,
                c_base=self.cBase,
            )
            action, action_idx, children_visit = select_action(
                1 - agent_id,
                None,
                selected_belief_states,
                [certainty for _ in range(len(selected_belief_states))],
                FOVs,
                None,
                env.steps - self.simulate_n_steps + sim_step,
                sim_envs,
                mcts[agent_id],
                "stop",
            )
            sim_actions.append(action)
            self.simulation["action_probs"].append(
                children_visit**self.beta / np.sum(children_visit**self.beta)
            )
            self.simulation["action_idx"].append(action_idx)  # for debugging
            self.simulation["world_states"].append(curr_sim_state[0])
            # transition for simulated agent
            curr_sim_state, certainty = sim_envs[1 - agent_id].transition(
                1 - agent_id,
                curr_sim_state[0],
                certainty,
                action,
                expected_action="stop",
            )
            curr_sim_state = [
                curr_sim_state.copy() for _ in range(self.args.n_particles)
            ]  # particles
            # update sim fov
            sim_envs[1 - agent_id].update_field_of_view()
            FOVs = [
                agent.field_of_view.copy() for agent in sim_envs[1 - agent_id].agents
            ]
        self.simulation["sim_actions"].append(sim_actions)
        self.simulation["sim_goals"].append(sim_goals)
        self.simulation["world_states"].append(curr_sim_state)
        print(sim_actions, env.actions[1 - agent_id][-self.simulate_n_steps :])
        # score = env.get_plan_score(sim_envs, agent_id, 'action_prob', simulation=self.simulation, self.simulate_n_steps)
        score = env.get_plan_score(
            sim_envs, agent_id, "pos_disparity", curr_sim_state[0][1 - agent_id]["pos"]
        )  # compare pos after simulate_n_steps actions
        self.simulation["plan_score"].append(score)
        print("plan score", score)

    def hiding_sim(self, certainty, FOVs, sim_envs, env, main_env):
        agent_id = np.where(self.args.full_obs)[0][0]
        sim_goals = []
        sim_actions = []
        score = 0
        # start with curr belief, simulate for the next 5 steps, progress based on transition with recommended action for chasing agent only
        curr_sim_state = [sim_envs[agent_id].get_state()]
        print("sim start pos", curr_sim_state[0][1 - agent_id]["pos"])
        accum_fov = FOVs[1 - agent_id].copy()
        mcts = [[None] * self.num_agents for _ in range(2)]
        for sim_step in range(self.simulate_n_steps):
            # TODO main_env.steps
            sim_envs[1 - agent_id].set_state(
                curr_sim_state[0][: self.num_agents],
                curr_sim_state[0][self.num_agents :],
                certainty,
                FOVs,
                main_env.steps,
            )
            (
                sim_envs[1 - agent_id].goals[1 - agent_id],
                selected_belief_states,
            ) = self.goal_decomposition_particles(
                curr_sim_state,
                certainty,
                FOVs,
                main_env.steps,
                sim_envs[1 - agent_id],
                1 - agent_id,
                self.goals[1 - agent_id],
            )
            sim_goals.append(sim_envs[1 - agent_id].goals[1 - agent_id])
            mcts[agent_id][1 - agent_id] = MCTS_Particle(
                agent_id=1 - agent_id,
                action_space=sim_envs[1 - agent_id].get_action_space(1 - agent_id),
                transition=sim_envs[1 - agent_id].transition,
                reward=sim_envs[1 - agent_id].get_reward_state,
                is_terminal=sim_envs[1 - agent_id].is_terminal,
                num_simulation=self.nb_simulations,
                max_rollout_steps=self.max_rollout_steps,
                c_init=self.cInit,
                c_base=self.cBase,
            )
            action, action_idx, children_visit = select_action(
                1 - agent_id,
                None,
                selected_belief_states,
                [certainty for _ in range(len(selected_belief_states))],
                FOVs,
                None,
                env.steps + sim_step,
                sim_envs,
                mcts[agent_id],
                "stop",
            )
            sim_actions.append(action)
            self.simulation["action_probs"].append(
                children_visit**self.beta / np.sum(children_visit**self.beta)
            )
            self.simulation["action_idx"].append(action_idx)  # for debugging
            self.simulation["world_states"].append(curr_sim_state[0])
            # transition for simulated agent
            curr_sim_state, certainty = sim_envs[1 - agent_id].transition(
                1 - agent_id,
                curr_sim_state[0],
                certainty,
                action,
                expected_action="stop",
            )
            curr_sim_state = [
                curr_sim_state.copy() for _ in range(self.args.n_particles)
            ]  # particles
            # update sim fov
            sim_envs[1 - agent_id].update_field_of_view()
            FOVs = [
                agent.field_of_view.copy() for agent in sim_envs[1 - agent_id].agents
            ]
            accum_fov += FOVs[1 - agent_id].copy()
            score += (
                1.0
                if sim_envs[1 - agent_id].in_field_of_view(
                    1 - agent_id, ("agent", agent_id)
                )
                else 0.0
            )
        self.simulation["sim_actions"].append(sim_actions)
        self.simulation["sim_goals"].append(sim_goals)
        self.simulation["world_states"].append(curr_sim_state)
        # score = env.get_plan_score(sim_envs, agent_id, 'visibility', curr_sim_state[0][1-agent_id]['pos'])
        self.simulation["accum_fovs"].append(accum_fov)
        # plt.imshow(accum_fov)
        # plt.show()
        self.simulation["plan_score"].append(score)
        print("plan score", score)

    def plan(self, nb_episodes, record=False):
        for episode_id in range(nb_episodes):
            print("episode_id", episode_id)
            self.episode_id = episode_id
            self.rollout(record, episode_id)
            gc.collect()

    def replay(
        self,
        arg,
        path,
        episode_id=0,
        video_name=None,
        agent_order=None,
        item_order=None,
        landmark_color_order=None,
        flip_env=None,
        rotate_env=False,
    ):
        p = Path(
            self.args.replay_dir
        )  # + '_{}_{}'.format(self.args.clip_interval[0], self.args.clip_interval[1]))
        if not p.is_dir():
            p.mkdir(parents=True)

        data = pickle.load(open(path, "rb"))

        self.replay_dir = str(p)

        # only render the first episode
        self.episode_id = episode_id

        flipped_goals_lms = self.goals.copy()
        # print(self.goals)
        # print(landmark_color_order)
        if landmark_color_order is not None:
            for agent_id, goal in enumerate(flipped_goals_lms):
                if "LMO" in goal or "LMA" in goal:
                    flipped_goals_lms[agent_id][2] = landmark_color_order[goal[2]]
        # flipped_goals_lms_colors = flipped_goals_lms.copy()
        flipped_goals_lms_colors = copy.deepcopy(flipped_goals_lms)
        # print(flipped_goals_lms_colors)
        if landmark_color_order is not None:
            for agent_id, goal in enumerate(flipped_goals_lms_colors):
                if "TE" in goal:
                    # print(goal[1], goal[2])
                    flipped_goals_lms_colors[agent_id][1] = agent_order[goal[1]] - 1
                    # print((agent_order+item_order)[goal[2]])
                    flipped_goals_lms_colors[agent_id][2] = (agent_order + item_order)[
                        goal[2]
                    ] - 1
                if "LMA" in goal:
                    flipped_goals_lms_colors[agent_id][1] = agent_order[goal[1]] - 1
                if "LMO" in goal:
                    flipped_goals_lms_colors[agent_id][1] = (
                        item_order[goal[1]] - 1 - len(agent_order)
                    )
        # print(flip_env)
        # print(agent_order)
        # print(item_order)
        # print(flipped_goals_lms_colors)

        if video_name is None:
            if self.args.flip_file_name:
                # TODO init pos & angles will not be updated in the name. only need goals (color+env), relations and strengths (color).
                video_name = "{}_{}{}_E{}_G{}_ST{}_SZ{}_P{}_A{}_C{}_AN{}_MCTS_L{}_R{}_PL{}_EL{}_{}_s{}_r{}_cI{}_cB{}_e{}".format(
                    "_".join(
                        [
                            "{}".format(f)
                            for f in [self.full_obs[idx - 1] for idx in agent_order]
                        ]
                    ),
                    "F" if self.action_type == "force" else "I",
                    8 if self.all_directions else 1,
                    str(self.env_id) + "." + str(flip_env)
                    if flip_env is not None
                    else self.env_id,
                    "_".join(
                        [
                            "{}".format(g)
                            for g in [
                                flipped_goals_lms_colors[idx - 1] for idx in agent_order
                            ]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(s)
                            for s in [self.strengths[idx - 1] for idx in agent_order]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(s)
                            for s in [
                                self.sizes[idx - 1] for idx in agent_order + item_order
                            ]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(p)
                            for p in [
                                self.init_positions[idx - 1]
                                for idx in agent_order + item_order
                            ]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(a)
                            for a in [
                                self.action_space_types[idx - 1] for idx in agent_order
                            ]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(c)
                            for c in [self.costs[idx - 1] for idx in agent_order]
                        ]
                    ),
                    "_".join(
                        [
                            "{}".format(d)
                            for d in [self.angles[idx - 1] for idx in agent_order]
                        ]
                    ),
                    self.max_episode_length,
                    "_".join(
                        [
                            "{}".format(float(a))
                            for a in [self.alpha[idx - 1] for idx in agent_order]
                        ]
                    ),
                    self.plan_length,
                    self.execute_length,
                    "_".join(
                        [
                            "{}".format(l)
                            for l in [self.levels[idx - 1] for idx in agent_order]
                        ]
                    ),
                    self.nb_simulations,
                    self.max_rollout_steps,
                    self.cInit,
                    self.cBase,
                    episode_id,
                )
            else:
                video_name = "{}_{}{}_E{}_G{}_ST{}_SZ{}_P{}_A{}_C{}_AN{}_MCTS_L{}_R{}_PL{}_EL{}_{}_s{}_r{}_cI{}_cB{}_e{}".format(
                    "_".join(["{}".format(f) for f in self.full_obs]),
                    "F" if self.action_type == "force" else "I",
                    8 if self.all_directions else 1,
                    self.env_id,
                    # '_'.join(['{}'.format(g) for g in self.goals]),
                    "_".join(["{}".format(g) for g in self.goals]),
                    # self.alpha[0], self.alpha[1],
                    "_".join(["{}".format(s) for s in self.strengths]),
                    "_".join(["{}".format(s) for s in self.sizes]),
                    "_".join(["{}".format(p) for p in self.init_positions]),
                    "_".join(["{}".format(a) for a in self.action_space_types]),
                    "_".join(["{}".format(c) for c in self.costs]),
                    "_".join(["{}".format(d) for d in self.angles]),
                    self.max_episode_length,
                    "_".join(["{}".format(float(a)) for a in self.alpha]),
                    # float(self.alpha[0]), float(self.alpha[1]),
                    self.plan_length,
                    self.execute_length,
                    "_".join(["{}".format(l) for l in self.levels]),
                    # self.levels[0], self.levels[1],
                    self.nb_simulations,
                    self.max_rollout_steps,
                    self.cInit,
                    self.cBase,
                    episode_id,
                )
            if self.args.save_date:
                # video_name = "D{}".format(datetime.now().strftime("%m%d%y_%H%M%S"))+video_name
                # video_name = "{}".format(arg['date'])+'_'+video_name #TODO
                video_name = "{}".format(arg["date"]) + video_name
            video_name = "/" + video_name
            print(video_name)

        self.main_env.seed()
        # self.env.set_clip_interval(self.args.clip_interval)
        self.main_env.setup(
            self.env_id,
            self.planning_agent,
            self.max_episode_length,
            record_path=self.replay_dir + "/" + video_name + ".mp4",
            agent_order=agent_order,
            item_order=item_order,
            landmark_color_order=landmark_color_order,
            flip_env=flip_env,
            rotate_env=rotate_env,
        )
        self.main_env.start(replay=True)

        # print(path)
        # print(data['actions'][0])
        data = pickle.load(open(path, "rb"))
        # print(data)
        plan = data["actions"]
        T = len(plan[0])
        # T = 10 #TODO
        rewards = []
        for t in range(T):
            for agent_id in range(self.num_agents):
                self.main_env.send_action(agent_id, plan[agent_id][t])
            self.main_env.step(replay=True)
            reward = [
                self.main_env.get_reward_state(
                    agent_id, self.main_env.get_state(), action=plan[agent_id][t]
                )
                for agent_id in range(self.num_agents)
            ]
            # print(t, plan[0][t], plan[1][t], reward)
            rewards.append(reward)

        # print('landmark_color_code',self.main_env.landmark_color_code)

        path = self.replay_dir + "/" + video_name + ".pik"
        if not self.main_env.human_simulation:  # TODO check this does not ruin anything
            pickle.dump(
                {
                    "arg": arg,
                    "trajectories": self.main_env.tmp_trajectories
                    if flip_env is not None
                    else self.main_env.trajectories,
                    "actions": self.main_env.actions,
                    "wall_segs": self.main_env.tmp_wall_segs
                    if flip_env is not None
                    else self.main_env.wall_segs,
                    "landmark_centers": self.main_env.landmark_centers,
                    "sizes": self.main_env.sizes_value,
                    "strengths": self.main_env.strengths_value,
                    "entity_color_code": self.main_env.entity_color_code,
                    "landmark_color_order": landmark_color_order,  # self.main_env.landmark_color_code,
                    "rotate_env": self.main_env.rotate_env,
                    "flip_env": self.main_env.flip_env,
                    "goals": self.goals,
                    "flipped_goals": flipped_goals_lms,
                    "modes": data["modes"],
                    "subgoals": data["subgoals"],
                    "rewards": rewards,
                    "all_modes": data["all_modes"],
                    "simulations": data["simulations"],
                    "sampled_beliefs": data["sampled_beliefs"],
                },
                open(path, "wb"),
            )

        # info_path = self.replay_dir + '/' + video_name[:-3] + 'txt'
        # f = open(info_path, 'w')
        # for entity_id in range(1, self.main_env.num_agents + self.main_env.num_items + 1):
        #     color_id = 0
        #     for i in range(3):
        #         if self.main_env.colors[entity_id][i] > 0:
        #             color_id = i
        #             break
        #     f.write('%d\n' % color_id)
        # f.close()

    def check_replay(self, arg, path, episode_id=0, video_name=None):
        p = Path(
            self.args.replay_dir
        )  # + '_{}_{}'.format(self.args.clip_interval[0], self.args.clip_interval[1]))
        if not p.is_dir():
            p.mkdir(parents=True)

        data_original = pickle.load(open(path, "rb"))

        self.replay_dir = str(p)

        # only render the first episode
        self.episode_id = episode_id

        if video_name is None:
            video_name = "{}_{}{}_E{}_G{}_ST{}_SZ{}_P{}_A{}_C{}_AN{}_MCTS_L{}_R{}_{}_PL{}_EL{}_{}_{}_s{}_r{}_cI{}_cB{}_e{}".format(
                "_".join(["{}".format(f) for f in self.full_obs]),
                "F" if self.action_type == "force" else "I",
                8 if self.all_directions else 1,
                self.env_id,
                "_".join(["{}".format(g) for g in self.goals]),
                # self.alpha[0], self.alpha[1],
                "_".join(["{}".format(s) for s in self.strengths]),
                "_".join(["{}".format(s) for s in self.sizes]),
                "_".join(["{}".format(p) for p in self.init_positions]),
                "_".join(["{}".format(a) for a in self.action_space_types]),
                "_".join(["{}".format(c) for c in self.costs]),
                "_".join(["{}".format(d) for d in self.angles]),
                self.max_episode_length,
                float(self.alpha[0]),
                float(self.alpha[1]),
                self.plan_length,
                self.execute_length,
                self.levels[0],
                self.levels[1],
                self.nb_simulations,
                self.max_rollout_steps,
                self.cInit,
                self.cBase,
                episode_id,
            )
            if self.args.save_date:
                video_name = "{}".format(arg["date"]) + video_name
            video_name = "/" + video_name

        path = self.replay_dir + "/" + video_name + ".pik"
        path = self.replay_dir + video_name + ".pik"
        print("path", path)
        data_replay = pickle.load(open(path, "rb"))

        print("episode id:", episode_id)
        for traj_original, traj_replay in zip(
            data_original["trajectories"][:4], data_replay["trajectories"][:4]
        ):
            d = get_dist(traj_original[-1][:2], traj_replay[-1][:2])
            print("final pos:", get_dist(traj_original[-1][:2], traj_replay[-1][:2]))
            # print('final vel:', get_dist(traj_original[-1][2:4], traj_replay[-1][2:4]))
            # print('final angle:', abs(traj_original[-1][-1] - traj_replay[-1][-1]))
        print("final rewards:", data_replay["rewards"][-1])
        return data_replay["rewards"][-1]

    def transition(
        self,
        agent_id,
        curr_state,
        action,
        expected_action,
        cInit,
        cBase,
        init_attached=False,
    ):
        """transition func (simulate one step)"""
        # if nb_steps % 10:
        # return self.sim_envs[self.planning_agent].transition(curr_state, action, nb_steps, cInit, cBase)
        other_agent_id = 1 - agent_id

        # self.sim_envs[agent_id].set_state(self.env.agents_pos, self.env.agents_vel, self.env.steps)
        # initState = self.sim_envs[agent_id].get_state(agent_id)
        # expected_action = 'upleft'#'None'
        # if not self.sim_envs[agent_id].is_terminal(initState, nb_steps):
        #     rootAction = None
        #     rootNode = Node(id={rootAction: initState},
        #                     num_visited=0,
        #                     sum_value=0,
        #                     is_expanded=True)
        #     currNode = rootNode
        #     currState = list(currNode.id.values())[0]
        #     # print(currStates[agent_id])
        #     # print(self.sim_envs[agent_id].is_terminal(currStates[agent_id], nb_steps))
        #     _, expected_action = self.mcts[agent_id](currNode, nb_steps)

        if self.env.num_items:
            """TODO: env_id"""
            self.sim_envs[agent_id]._setup_tmp(self.env_id, curr_state)
        else:
            self.sim_envs[agent_id]._setup_tmp(self.env_id, curr_state)
        self.sim_envs[agent_id]._send_action_tmp(agent_id, action)
        self.sim_envs[agent_id]._send_action_tmp(other_agent_id, expected_action)
        self.sim_envs[agent_id]._step_tmp()
        # print(curr_agents_pos, action, self.tmp_agents_pos)
        next_state = self.sim_envs[agent_id]._get_state_tmp()
        # print(curr_state, action, expected_action, next_state)
        return next_state
