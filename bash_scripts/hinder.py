import numpy as np
import random
import math
import json
import argparse

# envs + rooms:
paths = {}
# analyze each room to see what makes sense (could possibly just make a detour? - or not since 1 wall always closed..)
# possible bias - most doors leave 0,3 closed rooms. randomize landmark colors as well?
# small doors - only size 0 can pass. (other doors - all can pass):
# 4 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 8 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 17 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 10 - rms 0,1: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
key = (4, 8, 17, 10)
free_paths = ["12", "13", "23"]
paths[key] = free_paths
# 12 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 13 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 18 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
key = (12, 13, 18)
free_paths = ["03", "12"]
paths[key] = free_paths
# 14 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
# 19 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
key = (14, 19)
free_paths = ["12"]
paths[key] = free_paths
# 15 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
# 20 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
key = (15, 20)
free_paths = ["12"]
paths[key] = free_paths
# 21 - rms 0,3 + 2,3 + 1,2: all rooms are too small - agent a in any, item in other but notice helping agent should be small & anyehre or big & in same room as item + adjacent to agent a (0,1 no door).
key = (21,)
free_paths = []
paths[key] = free_paths
# open rooms
key = (0, 2, 6, 7, 9, 16)
free_paths = ["01", "02", "03", "12", "13", "23"]
paths[key] = free_paths


n_vids_per_env = 3  # 5
all_envs = [0, 2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 18
# symmetric positions
rm0pos = [0, 1, 10, 14]
rm1pos = [4, 5, 12, 16]
rm2pos = [6, 7, 13, 17]
rm3pos = [2, 3, 11, 15]
all_pos = rm0pos + rm1pos + rm2pos + rm3pos
pos_by_rm_dict = {
    "rm0pos": [0, 1, 10, 14],
    "rm1pos": [4, 5, 12, 16],
    "rm2pos": [6, 7, 13, 17],
    "rm3pos": [2, 3, 11, 15],
}
landmarks = [0, 1, 2, 3]
items = [0, 1]
agents = [0, 1]
# TODO sizes currently same for item and agent.in terms of passing through small doors
big_sizes = [1, 2, 3]  # [2] #[1, 2]
small_sizes = [0]  # [0, 1]
all_sizes = [0, 1, 2, 3]  # [0, 1, 2]
# big_sizes = small_sizes = all_sizes = [0]
# strengths
all_strengths = [1, 2, 3]  # [0, 1, 2, 3]
all_rooms = [0, 1, 2, 3]
all_goals = ["LMA", "TE", "LMO"]
alphas = {"agent0": [], "agent1": []}
sizes = {"agent0": [], "agent1": [], "item0": [], "item1": []}
strengths = {"agent0": [], "agent1": []}
angles = {"agent0": [], "agent1": []}
envs = np.repeat(all_envs, n_vids_per_env)
envs = np.random.permutation(envs)
goal_pres = []
goal_ag_it = []
goal_lm_it = []
goals_end = {"agent0": [], "agent1": []}
init_positions = {"agent0": [], "agent1": [], "item0": [], "item1": []}


def get_rm_path(rm1, rm2):
    rms = [rm1, rm2]
    return str(min(rms)) + str(max(rms))


for env in envs:
    # get room data for env
    for key in paths.keys():
        if env in key:
            free_rms = paths[key]
    hindered_id = 1
    hinder_id = 0
    goal_item_id = random.choice(items)
    item_id = 1 - goal_item_id
    goal_pre = random.choice(all_goals)  # ['LMA', 'LMO', 'GE']
    goal_item_rm = random.choice(landmarks)
    hinder_rm = random.choice(landmarks)

    if goal_pre == "LMO":
        # item
        goal_var1 = goal_item_id
        # lm
        goal_var2 = random.choice(
            list(set(landmarks) - set([goal_item_rm]))
        )  # lm != item rm
        hindered_rm = random.choice(landmarks)
        if get_rm_path(goal_item_rm, goal_var2) in free_rms:
            goal_item_size = random.choice(all_sizes)
        else:
            goal_item_size = random.choice(small_sizes)
        if get_rm_path(hinder_rm, goal_var2) in free_rms:
            hinder_size = random.choice(all_sizes)
        else:
            hinder_size = random.choice(small_sizes)
        if (
            get_rm_path(hindered_rm, goal_item_rm) in free_rms
            and get_rm_path(goal_item_rm, goal_var2) in free_rms
        ):
            hindered_size = random.choice(all_sizes)
        else:
            hindered_size = random.choice(small_sizes)

    if goal_pre == "LMA":
        # agent
        goal_var1 = hindered_id
        # lm
        goal_var2 = random.choice(landmarks)
        hindered_rm = random.choice(
            list(set(landmarks) - set([goal_var2]))
        )  # agent rm != lm rm
        goal_item_size = random.choice(all_sizes)
        if get_rm_path(hinder_rm, goal_var2) in free_rms:
            hinder_size = random.choice(all_sizes)
        else:
            hinder_size = random.choice(small_sizes)
        if get_rm_path(hindered_rm, goal_var2) in free_rms:
            hindered_size = random.choice(all_sizes)
        else:
            hindered_size = random.choice(small_sizes)

    if goal_pre == "TE":
        # agent
        goal_var1 = hindered_id
        # item
        goal_var2 = goal_item_id + len(agents)  # entity id
        hindered_rm = random.choice(
            list(set(landmarks) - set([goal_item_rm]))
        )  # agent rm != item rm
        goal_item_size = random.choice(all_sizes)
        if get_rm_path(hinder_rm, goal_item_rm) in free_rms:
            hinder_size = random.choice(all_sizes)
        else:
            hinder_size = random.choice(small_sizes)
        if get_rm_path(hindered_rm, goal_item_rm):
            hindered_size = random.choice(all_sizes)
        else:
            hindered_size = random.choice(small_sizes)

    alphas["agent" + str(hindered_id)].append(0)
    alphas["agent" + str(hinder_id)].append(-10)
    # depends on rooms
    sizes["agent" + str(hindered_id)].append(hindered_size)
    sizes["agent" + str(hinder_id)].append(hinder_size)
    sizes["item" + str(goal_item_id)].append(goal_item_size)
    sizes["item" + str(item_id)].append(random.choice(all_sizes))
    # 0 size goal item any strength can pick up
    strengths["agent" + str(hindered_id)].append(random.choice(all_strengths))
    strengths["agent" + str(hinder_id)].append(random.choice(all_strengths))
    # random
    angles["agent" + str(hindered_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    angles["agent" + str(hinder_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    # same goal (except hinder)
    goal_pres.append(goal_pre)
    goal_ag_it.append(goal_var1)
    goal_lm_it.append(goal_var2)
    goals_end["agent" + str(hindered_id)].append(1)
    goals_end["agent" + str(hinder_id)].append(-1)
    # depends on setup
    init_positions["agent" + str(hindered_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(hindered_rm) + "pos"])
    )
    init_positions["agent" + str(hinder_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(hinder_rm) + "pos"])
    )
    goal_item_pos = random.choice(pos_by_rm_dict["rm" + str(goal_item_rm) + "pos"])
    init_positions["item" + str(goal_item_id)].append(goal_item_pos)
    item_pos = random.choice(list(set(all_pos) - set([goal_item_pos])))
    init_positions["item" + str(item_id)].append(item_pos)


filename = "hinder.txt"
text_file = open(filename, "w")
text_file.write(" ".join(map(str, envs)))
print(" ".join(map(str, envs)))
for param_holder in [
    alphas,
    sizes,
    strengths,
    angles,
    goal_pres,
    goal_ag_it,
    goal_lm_it,
    goals_end,
    init_positions,
]:
    if type(param_holder) == dict:
        for key in sorted(param_holder.keys()):
            text_file.write("\n" + " ".join(map(str, param_holder[key])))
            print(" ".join(map(str, param_holder[key])))
    else:
        text_file.write("\n" + " ".join(map(str, param_holder)))
        print(" ".join(map(str, param_holder)))
text_file.write("\n")
text_file.close()
