import random
import copy
import numpy as np
from anytree import AnyNode as Node
from anytree import RenderTree
from tqdm import tqdm
import time


class MCTS_Particle:
    def __init__(
        self,
        agent_id,
        action_space,
        transition,  # attached,
        reward,
        is_terminal,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        seed=1,
    ):
        self.agent_id = agent_id
        self.transition = transition
        # self.attached = attached
        self.reward = reward
        self.is_terminal = is_terminal
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.action_space = list(action_space)
        self.rollout_policy = lambda state: random.choice(self.action_space)
        self.seed = seed
        random.seed(seed)
        self.nb_nodes = 0

    def run(self, curr_root, t, expected_action, dT=1):
        random.shuffle(self.action_space)
        particle_id = random.randint(0, len(curr_root.state_set) - 1)
        curr_root = self.expand(
            curr_root,
            curr_root.state_set[particle_id],
            curr_root.certainty_set[particle_id],
            t,
            expected_action,
        )
        for explore_step in tqdm(range(self.num_simulation)):
            # if explore_step % 1000 == 0 and self.num_simulation > 1000:
            #     print("simulation step:", explore_step, "out of", self.num_simulation)
            # start = time.time()
            # print("simulation step:", explore_step, "out of", self.num_simulation)
            curr_node = curr_root
            particle_id = random.randint(0, len(curr_node.state_set) - 1)
            curr_state = curr_node.state_set[particle_id]
            curr_certainty = curr_node.certainty_set[particle_id]
            node_path = [curr_node]
            state_path = [curr_state]
            certainty_path = [curr_certainty]

            tmp_t = t - 1
            while curr_node.is_expanded:
                next_node, next_state, next_certainty = self.select_child(
                    curr_node, curr_state, curr_certainty
                )
                next_node.state_set.append(copy.deepcopy(next_state))
                next_node.certainty_set.append(next_certainty.copy())

                node_path.append(next_node)
                state_path.append(next_state)
                certainty_path.append(next_certainty)

                curr_node = next_node
                curr_state = next_state
                curr_certainty = next_certainty
                tmp_t += 1

            leaf_node = self.expand(
                curr_node, curr_state, curr_certainty, tmp_t, expected_action
            )
            end = time.time()
            # print('before rollout:', end - start)
            # start = time.time()
            value = self.rollout(
                leaf_node, curr_state, curr_certainty, tmp_t, expected_action
            )
            # print(value)
            self.backup(self.agent_id, value, node_path, state_path, certainty_path)
            # end = time.time()
            # print('after rollout:', end - start)

        # next_root = None #self.select_next_root(curr_root)
        # action_taken = list(next_root.id.keys())[0]
        action_taken, children_visit, next_root, action_idx = self.select_next_root(
            curr_root
        )

        return next_root, action_taken, children_visit, action_idx

    def calculate_score(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        action_prior = child.action_prior

        if self_visit_count == 0:
            u_score = np.inf
            q_score = 0
        else:
            exploration_rate = (
                np.log((1 + parent_visit_count + self.c_base) / self.c_base)
                + self.c_init
            )
            u_score = (
                exploration_rate
                * action_prior
                * np.sqrt(parent_visit_count)
                / float(1 + self_visit_count)
            )
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score

    def select_child(self, curr_node, curr_state, curr_certainty):
        scores = [
            self.calculate_score(curr_node, child) for child in curr_node.children
        ]
        if len(scores) == 0:
            return None
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        next_state, next_certainty = self.transition(
            self.agent_id,
            curr_state,
            curr_certainty,
            selected_child.action,
            None,
            self.c_init,
            self.c_base,
        )
        return selected_child, next_state, next_certainty

    def get_action_prior(self):
        action_prior = {
            action: 1 / len(self.action_space) for action in self.action_space
        }
        return action_prior

    def expand(self, leaf_node, curr_state, curr_certainty, t, expected_action):
        if not self.is_terminal(curr_state, t):
            leaf_node.is_expanded = True
            leaf_node = self.initialize_children(
                leaf_node, curr_state, curr_certainty, expected_action
            )
        return leaf_node

    def rollout(self, leaf_node, curr_state, curr_certainty, t, expected_action):
        reached_terminal = False
        rewards = []
        sum_reward = 0
        next_certainty = [c.copy() for c in curr_certainty]

        for rollout_step in range(self.max_rollout_steps):
            action = self.rollout_policy(curr_state)
            expected_action = self.rollout_policy(curr_state)
            # print(action)
            # rewards.append(self.reward(self.agent_id, curr_state, action, t + rollout_step + 1, t + self.max_rollout_steps, goal_id=None,
            #                             prev_certainty=None, curr_certainty=None))
            rewards.append(
                self.reward(
                    self.agent_id,
                    curr_state,
                    action,
                    t + rollout_step + 1,
                    t + self.max_rollout_steps,
                    goal=None,
                    prev_certainty=curr_certainty,
                    curr_certainty=next_certainty,
                )
            )
            if self.is_terminal(
                curr_state, t + rollout_step + 1
            ):  # or t + rollout_step + 1 >= self.nb_steps:
                # f = open("rollout_terminal_heuristic_{}_{}.txt".format(dis, self.num_simulations), "a+")
                # print(1, file=f)
                reached_terminal = True
                break

            next_state, next_certainty = self.transition(
                self.agent_id,
                curr_state,
                curr_certainty,
                action,
                expected_action,
                self.c_init,
                self.c_base,
                update_certainty=False,
            )
            curr_state = next_state
            curr_certainty = next_certainty

        if rewards:
            sum_reward = rewards[-1]
            for r in reversed(rewards[:-1]):
                sum_reward = sum_reward * 0.95 + r
        return sum_reward

    def backup(self, agent_id, value, node_list, state_list, certainty_list):
        # if value > 0:
        #     print("=====================")
        cur_value = value
        t = len(node_list) - 1
        for node, state in zip(reversed(node_list), reversed(state_list)):
            action = node.action
            if t > 0:
                reward = self.reward(
                    agent_id,
                    state,
                    action,
                    0,
                    0,
                    prev_certainty=certainty_list[t - 1],
                    curr_certainty=certainty_list[t],
                )
            else:
                reward = self.reward(agent_id, state, action, 0, 0)
            cur_value = cur_value * 0.95 + reward
            node.sum_value += cur_value
            node.num_visited += 1
            t -= 1
            # if value > 0:
            #     print(node)

    def select_next_root(self, curr_root):
        children_visit = [child.num_visited for child in curr_root.children]
        print(
            "children_visit:",
            list(zip(children_visit, [child.action for child in curr_root.children])),
        )
        maxIndex = np.argwhere(children_visit == np.max(children_visit)).flatten()
        selected_child_index = random.choice(maxIndex)
        action_prob = children_visit[selected_child_index] / np.sum(children_visit)
        action = curr_root.children[selected_child_index].action
        return (
            action,
            children_visit,
            curr_root.children[selected_child_index],
            selected_child_index,
        )  # curr_root.children[selected_child_index]=next_root

    def initialize_children(self, node, state, certainty, expected_action):
        initActionPrior = self.get_action_prior()

        for action in self.action_space:
            next_state, next_certainty = self.transition(
                self.agent_id,
                state,
                certainty,
                action,
                expected_action,
                self.c_init,
                self.c_base,
            )
            # print("child initialized", state, action, nextState)
            Node(
                parent=node,
                id=self.nb_nodes,
                action=action,
                state_set=[next_state],
                certainty_set=[next_certainty],
                num_visited=0,
                sum_value=0,
                action_prior=initActionPrior[action],
                is_expanded=False,
            )
            self.nb_nodes += 1

        return node
