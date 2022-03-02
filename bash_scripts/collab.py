import numpy as np
import random
import math
import json
import argparse

# weak agent scenarios
# 2 weak (what's weak?), far away from item (different room than item).
# need to make sure they can enter the rooms they're trying to get to.
# envs that all can pass with size + strength combinations:
# (0 2 3 6 7 9 16) - for any size. others only for small (0,1).

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


n_vids_per_env = 3
all_envs = [0, 2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 18
# symmetric positions
rm0pos = [0, 1, 10, 14]
rm1pos = [4, 5, 12, 16]
rm2pos = [6, 7, 13, 17]
rm3pos = [2, 3, 11, 15]
POS = [
    (16 - 8, 12 + 8),  # 0
    (16 - 8, 12 + 2),  # 1
    (16 - 2, 12 + 4),  # 10
    (16 - 2, 12 + 8),  # 14
    (16 + 8, 12 + 8),  # 4
    (16 + 8, 12 + 2),  # 5
    (16 + 2, 12 + 4),  # 12
    (16 + 2, 12 + 8),  # 16
    (16 + 8, 12 - 2),  # 6
    (16 + 8, 12 - 8),  # 7
    (16 + 2, 12 - 4),  # 13
    (16 + 2, 12 - 8),  # 17
    (16 - 8, 12 - 2),  # 2
    (16 - 8, 12 - 8),  # 3
    (16 - 2, 12 - 4),  # 11
    (16 - 2, 12 - 8),  # 15
]
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
# low_strength = [0, 1]
# high_strength = [2, 3]
all_strengths = [0, 1, 2, 3]  # [0, 1, 2, 3]
all_rooms = [0, 1, 2, 3]
alphas = {"agent0": [], "agent1": []}
sizes = {"agent0": [], "agent1": [], "item0": [], "item1": []}
strengths = {"agent0": [], "agent1": []}
angles = {"agent0": [], "agent1": []}
envs = np.repeat(all_envs, n_vids_per_env)
envs = np.random.permutation(envs)
goal_pres = []
goal_ag_it = []
goal_lm_it = []
goals_end = []
init_positions = {"agent0": [], "agent1": [], "item0": [], "item1": []}


def get_rm_path(rm1, rm2):
    rms = [rm1, rm2]
    return str(min(rms)) + str(max(rms))


for env in envs:
    # get room data for env
    for key in paths.keys():
        if env in key:
            free_rms = paths[key]
    colab1_id = random.choice(agents)
    colab2_id = 1 - colab1_id
    goal_item_id = random.choice(items)
    item_id = 1 - goal_item_id
    goal_pre = "LMO"
    goal_item_rm = random.choice(landmarks)
    goal_lm = random.choice(list(set(landmarks) - set([goal_item_rm])))  # lm != item rm
    # #agents != item rm
    # colab1_rm = random.choice(list(set(landmarks)-set([goal_item_rm])))
    # colab2_rm = random.choice(list(set(landmarks)-set([goal_item_rm])))

    if env in [10, 15, 20]:
        rm_order = [0, 1, 2, 3]
    else:
        rm_order = [0, 3, 2, 1]
    goal_item_rm_idx = np.where(np.array(rm_order) == goal_item_rm)[0]
    colab1_rm = random.choice(list(set(landmarks) - set([goal_item_rm])))
    colab1_rm_idx = np.where(np.array(rm_order) == colab1_rm)[0]
    if goal_item_rm_idx == 1 and colab1_rm_idx in [0, 2]:
        colab2_rm = random.choice([rm_order[0], rm_order[2]])
    elif goal_item_rm_idx == 2 and colab1_rm_idx in [1, 3]:
        colab2_rm = random.choice([rm_order[1], rm_order[3]])
    else:
        colab2_rm = colab1_rm

    if get_rm_path(goal_item_rm, goal_lm) in free_rms:
        goal_item_size = random.choice(all_sizes)
    else:
        goal_item_size = random.choice(small_sizes)
    if (
        get_rm_path(colab1_rm, goal_item_rm) in free_rms
        and get_rm_path(goal_item_rm, goal_lm) in free_rms
    ):
        colab1_size = random.choice(all_sizes)
    else:
        colab1_size = random.choice(small_sizes)
    if (
        get_rm_path(colab2_rm, goal_item_rm) in free_rms
        and get_rm_path(goal_item_rm, goal_lm) in free_rms
    ):
        colab2_size = random.choice(all_sizes)
    else:
        colab2_size = random.choice(small_sizes)

    alphas["agent" + str(colab1_id)].append(0)
    alphas["agent" + str(colab2_id)].append(0)  # 10
    # depends on rooms
    sizes["agent" + str(colab1_id)].append(colab1_size)
    sizes["agent" + str(colab2_id)].append(colab2_size)
    sizes["item" + str(goal_item_id)].append(goal_item_size)
    sizes["item" + str(item_id)].append(random.choice(all_sizes))
    # 0 size goal item any strength can pick up
    st1 = random.choice(all_strengths)
    strengths["agent" + str(colab1_id)].append(st1)
    strengths["agent" + str(colab2_id)].append(st1)
    # random
    angles["agent" + str(colab1_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    angles["agent" + str(colab2_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    # same goal (except hinder)
    goal_pres.append(goal_pre)
    goal_ag_it.append(goal_item_id)
    goal_lm_it.append(goal_lm)
    goals_end.append(1)
    # depends on setup
    init_positions["agent" + str(colab1_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(colab1_rm) + "pos"])
    )
    init_positions["agent" + str(colab2_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(colab2_rm) + "pos"])
    )
    goal_item_pos = random.choice(pos_by_rm_dict["rm" + str(goal_item_rm) + "pos"])
    init_positions["item" + str(goal_item_id)].append(goal_item_pos)
    item_pos = random.choice(list(set(all_pos) - set([goal_item_pos])))
    init_positions["item" + str(item_id)].append(item_pos)


filename = "collab.txt"
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
