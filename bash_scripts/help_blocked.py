import numpy as np
import random
import math
import json
import argparse


# small door scenarios
#  agent too big to get to item (size 1,2 + small room envs). in target rm/lm.
#  other: (a) small+anywhere (b) in item room + any size

# envs + rooms:
small_room_combs = {}
# analyze each room to see what makes sense (could possibly just make a detour? - or not since 1 wall always closed..)
# possible bias - most doors leave 0,3 closed rooms. randomize landmark colors as well?
# small doors - only size 0 can pass. (other doors - all can pass):
# 4 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 8 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 17 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 10 - rms 0,1: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 4,8,17 door 0; 10 door 1
key = (4, 8, 17)
room_combs = [[[0], [3]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0]}
key = (10,)
room_combs = [[[0], [1]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0]}
# 12 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 13 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 18 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# door 0
key = (12, 13, 18)
room_combs = [[[3], [2]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0]}
# 14 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
# 19 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
# door 0
key = (14, 19)
room_combs = [[[0], [3]], [[3], [2]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0, 0]}
# 15 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
# 20 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
# door 1
key = (15, 20)
room_combs = [[[0], [1]], [[2], [3]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0, 0]}
# 21 - rms 0,3 + 2,3 + 1,2: all rooms are too small - agent a in any, item in other but notice helping agent should be small & anyehre or big & in same room as item + adjacent to agent a (0,1 no door).
# door 0
key = (21,)
room_combs = [[[0], [3]], [[3], [2]], [[2], [1]]]
small_room_combs[key] = {"room_combs": room_combs, "small_helper": [0, 0, 0]}


n_vids_per_small_door_env = 1  # 6
small_door_envs = [4, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21]  # 12 envs
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
# TODO sizes currently same for item and agent. in terms of passing through small doors
big_sizes = [1, 2, 3]  # [2] #[1, 2]
small_sizes = [0]  # [0, 1]
all_sizes = [0, 1, 2, 3]  # [0, 1, 2]
# strengths
all_strengths = [1, 2, 3]  # [0, 1, 2, 3]
all_rooms = [0, 1, 2, 3]

goals = ["LMO"]
# goals = ['TE']

alphas = {"agent0": [], "agent1": []}
sizes = {"agent0": [], "agent1": [], "item0": [], "item1": []}
strengths = {"agent0": [], "agent1": []}
angles = {"agent0": [], "agent1": []}
envs = np.repeat(small_door_envs, n_vids_per_small_door_env)
envs = np.random.permutation(envs)
goals_pres = []
goals_items = {"agent0": [], "agent1": []}  # ['LMO', item, lm, 1]
goals_landmarks = {"agent0": [], "agent1": []}  # ['LMO', item, lm, 1]
init_positions = {"agent0": [], "agent1": [], "item0": [], "item1": []}

for env in envs:
    goal_pre = random.choice(goals)
    helpee_id = 1
    helpee_size = random.choice(big_sizes)
    helper_id = 0
    goal_item_id = random.choice(items)
    item_id = 1 - goal_item_id
    # get room data for env
    for key in small_room_combs.keys():
        if env in key:
            param_options = small_room_combs[key]
    # get room combos for helpee & goal item
    n_combs = len(param_options["room_combs"])
    # choose 1 combo
    comb_idx = random.choice(range(n_combs))
    lm_item_rms = param_options["room_combs"][comb_idx]
    is_helper_small = param_options["small_helper"][comb_idx]

    # helper on either side
    # #choose room for lm & goal item out of combo, helper on one (if small anywhere), helpee on other
    # goal_lm_idx = random.choice([0,1])
    # goal_item_idx = 1 - goal_lm_idx
    # goal_lm_rm = random.choice(lm_item_rms[goal_lm_idx])
    # goal_item_rm = random.choice(lm_item_rms[goal_item_idx])
    # #choose params for helper
    # # agent a size [1,2] - helpee
    # # agent b size [0] + anywhere -or- size [1,2] + access to item - helper
    # if is_helper_small:
    #     helper_size = random.choice(small_sizes)
    #     helper_rm = random.choice(all_rooms)
    #     helpee_rm = random.choice(lm_item_rms[0]+lm_item_rms[1])
    # else:
    #     # helper_size = random.choice(big_sizes)
    #     helper_size = random.choice(all_sizes)
    #     helper_rm_idx = random.choice([0,1])
    #     helpee_rm_idx = 1-helper_rm_idx
    #     helper_rm = random.choice(lm_item_rms[helper_rm_idx])
    #     helpee_rm = random.choice(lm_item_rms[helpee_rm_idx])

    # helper on item side
    # choose room for helpee & goal item out of combo
    helpee_rm_idx = random.choice([0, 1])
    helpee_rms = lm_item_rms[helpee_rm_idx]
    goal_item_idx = 1 - helpee_rm_idx
    goal_item_rms = lm_item_rms[goal_item_idx]
    helpee_rm = random.choice(helpee_rms)
    goal_item_rm = random.choice(goal_item_rms)
    # choose params for helper
    # agent a size [1,2] - helpee
    # agent b size [0] + anywhere -or- size [1,2] + access to item - helper
    if is_helper_small:
        helper_size = random.choice(small_sizes)
        helper_rm = random.choice(all_rooms)
    else:
        # helper_size = random.choice(big_sizes)
        helper_size = random.choice(all_sizes)
        helper_rm = random.choice(goal_item_rms)
    goal_lm_rm = helpee_rm

    if goal_pre == "LMO":
        g1 = goal_item_id
        g2 = goal_lm_rm
    elif goal_pre == "TE":
        g1 = helpee_id
        g2 = goal_item_id + len(agents)

    # TODO is neutral with same goal enough? - no get_subgoals_put_help..
    alphas["agent" + str(helpee_id)].append(0)
    alphas["agent" + str(helper_id)].append(10)
    # helpee is big, goal item small, helper depends on rooms
    sizes["agent" + str(helpee_id)].append(random.choice(big_sizes))
    sizes["agent" + str(helper_id)].append(helper_size)
    sizes["item" + str(goal_item_id)].append(random.choice(small_sizes))
    sizes["item" + str(item_id)].append(random.choice(all_sizes))
    # 0 size goal item any strength can pick up
    strengths["agent" + str(helpee_id)].append(random.choice(all_strengths))
    strengths["agent" + str(helper_id)].append(random.choice(all_strengths))
    # random
    angles["agent" + str(helpee_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    angles["agent" + str(helper_id)].append(
        round(math.radians(np.random.uniform(-360, 360)), 2)
    )
    # same goal. lm depends on setup
    goals_pres.append(goal_pre)
    goals_items["agent" + str(helpee_id)].append(g1)  # goal_item_id
    goals_items["agent" + str(helper_id)].append(g1)
    goals_landmarks["agent" + str(helpee_id)].append(g2)  # goal_lm_rm
    goals_landmarks["agent" + str(helper_id)].append(g2)
    # depends on setup
    init_positions["agent" + str(helpee_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(helpee_rm) + "pos"])
    )
    init_positions["agent" + str(helper_id)].append(
        random.choice(pos_by_rm_dict["rm" + str(helper_rm) + "pos"])
    )
    goal_item_pos = random.choice(pos_by_rm_dict["rm" + str(goal_item_rm) + "pos"])
    init_positions["item" + str(goal_item_id)].append(goal_item_pos)
    item_pos = random.choice(list(set(all_pos) - set([goal_item_pos])))
    init_positions["item" + str(item_id)].append(item_pos)


filename = "help_blocked.txt"
text_file = open(filename, "w")
text_file.write(" ".join(map(str, envs)))
print(" ".join(map(str, envs)))
for param_holder in [
    alphas,
    sizes,
    strengths,
    angles,
    goals_pres,
    goals_items,
    goals_landmarks,
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
