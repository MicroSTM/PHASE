import numpy as np
import random
import math
import json
import argparse

# small door scenarios
#  agent needs to get out of room, big item blocking

# envs + rooms:
small_room_combs = {}
# counter clockwise
# block door 0 from rm 0: 14, rm 1: 16
# block door 1 from rm 3: 2, rm 0: 1
# block door 2 from rm 2: 17, rm 3: 15
# block door 3 from rm 1: 5, rm 2: 6
# blocking_item_pos = {0:[14,16], 1:[1,2], 2:[15, 17], 3:[5,6]}
# analyze each room to see what makes sense (could possibly just make a detour? - or not since 1 wall always closed..)
# possible bias - most doors leave 0,3 closed rooms. randomize landmark colors as well?
# small doors - only size 0 can pass. (other doors - all can pass):
# 4 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 8 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 17 - rms 0,3: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
# 10 - rms 0,1: agent a in rm 0 + item 1,2,3 -or- agent a 1,2,3 + item in rm 0
key = (4, 8, 17)
room_combs = [[[0], [3]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0],
    "ball_loc": [[1, 2]],
}
key = (10,)
room_combs = [[[0], [1]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0],
    "ball_loc": [[14, 16]],
}
# 12 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 13 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
# 18 - rms 2,3: agent a in rm 0,3 + item 1,2 -or- agent a 1,2 + item in rm 0,3
key = (12, 13, 18)
room_combs = [[[3], [2]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0],
    "ball_loc": [[15, 17]],
}
# 14 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
# 19 - rms 0,3 + 2,3: agent a in 0 + item 3 or 1,2 -or- agent in 3 + item 0 or 1,2 -or- agent in 1,2 + item 0,3
key = (14, 19)
room_combs = [[[0], [3]], [[3], [2]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0, 0],
    "ball_loc": [[1, 2], [15, 17]],
}
# 15 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
# 20 - rms 0,1 + 2,3: agent a in 0 + item 1,2,3 -or- agent in 3 + item 0,1,2 -or- agent in 1,2 + item 0 or 3
key = (15, 20)
room_combs = [[[0], [1]], [[2], [3]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0, 0],
    "ball_loc": [[14, 16], [17, 15]],
}
# 21 - rms 0,3 + 2,3 + 1,2: all rooms are too small - agent a in any, item in other but notice helping agent should be small & anyehre or big & in same room as item + adjacent to agent a (0,1 no door).
key = (21,)
room_combs = [[[0], [3]], [[3], [2]], [[2], [1]]]
small_room_combs[key] = {
    "room_combs": room_combs,
    "small_helper": [0, 0, 0],
    "ball_loc": [[1, 2], [15, 17], [6, 5]],
}


n_vids_per_small_door_env = 1  # 6
small_door_envs = [4, 8, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21]  # 12 envs
# symmetric positions
rm0pos = [0, 1, 10, 14]
rm1pos = [4, 5, 12, 16]
rm2pos = [6, 7, 13, 17]
rm3pos = [2, 3, 11, 15]
all_pos = rm0pos + rm1pos + rm2pos + rm3pos
all_rooms = [0, 1, 2, 3]
pos_by_rm_dict = {
    "rm0pos": [0, 1, 10, 14],
    "rm1pos": [4, 5, 12, 16],
    "rm2pos": [6, 7, 13, 17],
    "rm3pos": [2, 3, 11, 15],
}
landmarks = [0, 1, 2, 3]
items = [0, 1]
agents = [0, 1]

big_sizes = [1, 2, 3]  # [2] #[1, 2]
small_sizes = [0]  # [0, 1]
all_sizes = [0, 1, 2, 3]  # [0, 1, 2]
item_big_sizes = [2]
# big_sizes = small_sizes = all_sizes = [0]
# strengths
low_strength = [0, 1]
high_strength = [3]
all_strengths = [0, 1, 2, 3]

all_goals = ["LMA"]
# all_goals = ['LMA', 'LMO']
# all_goals = ['TE']

alphas = {"agent0": [], "agent1": []}
sizes = {"agent0": [], "agent1": [], "item0": [], "item1": []}
strengths = {"agent0": [], "agent1": []}
angles = {"agent0": [], "agent1": []}
envs = np.repeat(small_door_envs, n_vids_per_small_door_env)
envs = np.random.permutation(envs)
goal_pres = []
goal_ag_it = []
goal_lm_it = []
goals_end = {"agent0": [], "agent1": []}
init_positions = {"agent0": [], "agent1": [], "item0": [], "item1": []}

# helpee in room with door blocked, goal - LMA (lm), LMO (lm), TE (item that can pick) on other side
# weak strength, small needs to get through door
# helper on other side, open door
# have helper be 10, -10, 0 as well so scenario not biased

# for mode in ['help', 'neutral']:
for mode in ["help"]:
    for env in envs:
        helpee_id = 1
        helpee_size = random.choice(small_sizes)
        helper_id = 0
        # get room data for env
        for key in small_room_combs.keys():
            if env in key:
                param_options = small_room_combs[key]
        # get room combos for helpee & goal item
        n_combs = len(param_options["room_combs"])
        # choose 1 combo
        comb_idx = random.choice(range(n_combs))
        helpee_item_rms = param_options["room_combs"][comb_idx]
        blocking_locations = param_options["ball_loc"][comb_idx]
        # choose room for helpee & goal item out of combo
        helpee_rm_idx = random.choice([0, 1])
        helpee_rms = helpee_item_rms[helpee_rm_idx]
        goal_item_lm_idx = 1 - helpee_rm_idx
        goal_item_lm_rms = helpee_item_rms[goal_item_lm_idx]
        helpee_rm = random.choice(helpee_rms)
        goal_item_lm_rm = random.choice(goal_item_lm_rms)
        helper_size = random.choice(all_sizes)
        helper_rm = random.choice(goal_item_lm_rms)
        # if helper_size in small_sizes:
        #     blocking_item_pos = random.choice(blocking_locations)
        # else:
        #     blocking_item_pos = blocking_locations[goal_item_lm_idx]
        blocking_item_pos = blocking_locations[goal_item_lm_idx]

        # item 0 for blocking - near door - on helper side if big helper, on any side if small helper
        # item 1 for LMO - anywhere, helpee side: small. item for TE - helper side, any size
        blocking_item_id = 0
        item_id = 1
        goal_pre = random.choice(all_goals)  #'LMA'
        if goal_pre == "LMA":
            item_rm = random.choice(all_rooms)
            goal_var1 = helpee_id
            goal_var2 = random.choice(goal_item_lm_rms)
        elif goal_pre == "LMO":
            goal_var1 = item_id
            # either lm or item or both on other side
            op = random.choice([1, 2, 3])
            if op == 1:
                item_rm = random.choice(helpee_rms)
                goal_var2 = random.choice(goal_item_lm_rms)
            elif op == 2:
                item_rm = random.choice(goal_item_lm_rms)
                goal_var2 = random.choice(helpee_rms)
            elif op == 3:
                item_rm = random.choice(goal_item_lm_rms)
                goal_var2 = random.choice(goal_item_lm_rms)
        elif goal_pre == "TE":
            item_rm = random.choice(goal_item_lm_rms)
            goal_var1 = helpee_id
            goal_var2 = len(agents) + item_id

        if goal_pre == "LMO" and (item_rm in helpee_rms or goal_var2 in helpee_rms):
            item_size = random.choice(small_sizes)
        else:
            item_size = random.choice(all_sizes)

        alphas["agent" + str(helpee_id)].append(0)
        if mode == "help":
            alphas["agent" + str(helper_id)].append(10)
        elif mode == "hinder":
            alphas["agent" + str(helper_id)].append(-10)
        elif mode == "neutral":
            alphas["agent" + str(helper_id)].append(0)

        # helpee is big, goal item small, helper depends on rooms
        sizes["agent" + str(helpee_id)].append(random.choice(small_sizes))
        sizes["agent" + str(helper_id)].append(helper_size)
        sizes["item" + str(blocking_item_id)].append(random.choice(item_big_sizes))
        sizes["item" + str(item_id)].append(item_size)
        # 0 size goal item any strength can pick up
        strengths["agent" + str(helpee_id)].append(random.choice(low_strength))
        strengths["agent" + str(helper_id)].append(random.choice(high_strength))
        # random
        angles["agent" + str(helpee_id)].append(
            round(math.radians(np.random.uniform(-360, 360)), 2)
        )
        angles["agent" + str(helper_id)].append(
            round(math.radians(np.random.uniform(-360, 360)), 2)
        )
        # same goal. lm depends on setup
        goal_pres.append(goal_pre)
        goal_ag_it.append(goal_var1)
        goal_lm_it.append(goal_var2)
        goals_end["agent" + str(helpee_id)].append(1)
        if mode == "help" or mode == "neutral":
            goals_end["agent" + str(helper_id)].append(1)
        elif mode == "hinder":
            goals_end["agent" + str(helper_id)].append(-1)
        # depends on setup
        init_positions["agent" + str(helpee_id)].append(
            random.choice(pos_by_rm_dict["rm" + str(helpee_rm) + "pos"])
        )
        init_positions["agent" + str(helper_id)].append(
            random.choice(pos_by_rm_dict["rm" + str(helper_rm) + "pos"])
        )
        init_positions["item" + str(blocking_item_id)].append(blocking_item_pos)
        item_pos = list(
            set(pos_by_rm_dict["rm" + str(item_rm) + "pos"]) - set([blocking_item_pos])
        )
        init_positions["item" + str(item_id)].append(random.choice(item_pos))

envs = list(envs) * 3
filename = "help_move.txt"
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
