import numpy as np
import random
import math
import json

paths = {}
paths[(4, 8, 17, 10)] = ['12', '13', '23']
paths[(12, 13, 18)] = ['03', '12']
paths[(14, 19)] = ['12']
paths[(15, 20)] = ['12']
paths[(21,)] = []
paths[(0, 2, 6, 7, 9, 16)] = ['01','02','03','12','13','23']


n_vids_per_env = 1
all_envs = [0, 2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] #18

rm0pos = [0, 1, 10, 14]
rm1pos = [4, 5, 12, 16]
rm2pos = [6, 7, 13, 17]
rm3pos = [2, 3, 11, 15]
pos_by_rm_dict = {'rm0pos':rm0pos, 'rm1pos':rm1pos, 'rm2pos':rm2pos, 'rm3pos':rm3pos}
all_pos = rm0pos + rm1pos + rm2pos + rm3pos
landmarks = [0,1,2,3]
small_sizes = [0]
all_sizes = [0,1,2,3]
all_strengths = [0,1,2,3]

# env_id = 0
# goals = ['LMO', 'LMA', 'TE']
# goals = ['LMO']*3 + ['LMA']*3 + ['TE']*2
goals = ['LMO', 'LMA']
# goal_opts1 = {'LMO':[0,1], 'LMA':[0,1], 'TE':[0,1]}
# goal_opts2 = {'LMO':landmarks, 'LMA':landmarks, 'TE':[2,3]}

envs = np.repeat(all_envs, n_vids_per_env)
envs = np.random.permutation(envs)
alphas = {'agent0':[], 'agent1':[]}
sizes = {'agent0':[], 'agent1':[], 'item0':[], 'item1':[]}
strengths = {'agent0':[], 'agent1':[]}
angles = {'agent0':[], 'agent1':[]}
goals0 = {'agent0':[], 'agent1':[]}
goals1 = {'agent0':[], 'agent1':[]}
goals2 = {'agent0':[], 'agent1':[]}
goals3 = {'agent0':[], 'agent1':[]}
init_positions = {'agent0':[], 'agent1':[], 'item0':[], 'item1':[]}

def get_rm_path(rm1,rm2):
    rms = [rm1,rm2]
    return str(min(rms))+str(max(rms))

for env_id in envs:
    for key in paths.keys():
        if env_id in key:
            free_rms = paths[key]

    g0_ag0 = random.choice(goals) #always agent 0, item 0
    g0_ag1 = random.choice(goals) #always agent 1, item 1

    if g0_ag0 == 'LMO':
        #item can't be on target lm
        g2_ag0 = random.choice(landmarks)
        it0_rm = random.choice(list(set(landmarks)-set([g2_ag0])))
        ag0_rm = random.choice(landmarks)
        if get_rm_path(it0_rm, g2_ag0) in free_rms:
            it0_sz = random.choice(all_sizes)
        else:
            it0_sz = random.choice(small_sizes)
        if get_rm_path(g2_ag0, it0_rm) in free_rms and \
            get_rm_path(it0_rm, g2_ag0) in free_rms:
            ag0_sz = random.choice(all_sizes)
        else:
            ag0_sz = random.choice(small_sizes)
    if g0_ag0 == 'LMA':
        #agent not on target lm
        g2_ag0 = random.choice(landmarks)
        ag0_rm = random.choice(list(set(landmarks)-set([g2_ag0])))
        it0_rm = random.choice(landmarks)
        it0_sz = random.choice(all_sizes)
        if get_rm_path(ag0_rm, g2_ag0) in free_rms:
            ag0_sz = random.choice(all_sizes)
        else:
            ag0_sz = random.choice(small_sizes)
    if g0_ag0 == 'TE':
        #agent not in same rm as item
        # g2_ag0 = 2
        # it0_rm = random.choice(landmarks)
        # ag0_rm = random.choice(list(set(landmarks)-set([it0_rm])))
        # if get_rm_path(ag0_rm, it0_rm) in free_rms:
        #     ag0_sz = random.choice(all_sizes)
        # else:
        #     ag0_sz = random.choice(small_sizes)
        #touch agent
        g2_ag0 = 1
        ag0_rm = random.choice(landmarks)
        it0_rm = random.choice(landmarks)
        it0_sz = random.choice(all_sizes)
        ag0_sz = random.choice(all_sizes)


    if g0_ag1 == 'LMO':
        g2_ag1 = random.choice(landmarks)
        it1_rm = random.choice(list(set(landmarks)-set([g2_ag1])))
        ag1_rm = random.choice(list(set(landmarks)-set([it1_rm])))
        if get_rm_path(it1_rm, g2_ag1) in free_rms:
            it1_sz = random.choice(all_sizes)
        else:
            it1_sz = random.choice(small_sizes)
        if get_rm_path(g2_ag1, it1_rm) in free_rms and \
            get_rm_path(it1_rm, g2_ag1) in free_rms:
            ag1_sz = random.choice(all_sizes)
        else:
            ag1_sz = random.choice(small_sizes)
    if g0_ag1 == 'LMA':
        g2_ag1 = random.choice(landmarks)
        ag1_rm = random.choice(list(set(landmarks)-set([g2_ag1])))
        it1_rm = random.choice(landmarks)
        it1_sz = random.choice(all_sizes)
        if get_rm_path(ag1_rm, g2_ag1) in free_rms:
            ag1_sz = random.choice(all_sizes)
        else:
            ag1_sz = random.choice(small_sizes)
    if g0_ag1 == 'TE':
        # g2_ag1 = 3
        # it1_rm = random.choice(landmarks)
        # ag1_rm = random.choice(list(set(landmarks)-set([it1_rm])))
        #touch agent
        g2_ag1 = 0
        ag1_rm = random.choice(list(set(landmarks)-set([ag0_rm])))
        it1_rm = random.choice(landmarks)
        it1_sz = random.choice(all_sizes)
        if get_rm_path(ag1_rm, ag0_rm) in free_rms:
            ag1_sz = random.choice(all_sizes)
        else:
            ag1_sz = random.choice(small_sizes)

    #double check
    if g0_ag0 == 'TE':
        ag0_rm = random.choice(list(set(landmarks)-set([ag1_rm])))
        if get_rm_path(ag0_rm, ag1_rm) in free_rms:
            ag0_sz = random.choice(all_sizes)
            ag1_sz = random.choice(all_sizes)
        else:
            ag0_sz = random.choice(small_sizes)
            ag1_sz = random.choice(small_sizes)

    # envs.append(env_id)
    #neutral
    alphas['agent0'].append(0)
    alphas['agent1'].append(0)
    #size
    sizes['agent0'].append(ag0_sz)
    sizes['agent1'].append(ag1_sz)
    sizes['item0'].append(it0_sz)
    sizes['item1'].append(it1_sz)
    #strength
    strengths['agent0'].append(random.choice(all_strengths))
    strengths['agent1'].append(random.choice(all_strengths))
    #random
    angles['agent0'].append(round(math.radians(np.random.uniform(-360,360)),2))
    angles['agent1'].append(round(math.radians(np.random.uniform(-360,360)),2))
    #goals
    goals0['agent0'].append(g0_ag0)
    goals0['agent1'].append(g0_ag1)
    goals1['agent0'].append(0)
    goals1['agent1'].append(1)
    goals2['agent0'].append(g2_ag0)
    goals2['agent1'].append(g2_ag1)
    goals3['agent0'].append(1)
    goals3['agent1'].append(1)
    #init - not overlapping, goal not true at init
    init_positions['agent0'].append(random.choice(pos_by_rm_dict['rm'+str(ag0_rm)+'pos']))
    init_positions['agent1'].append(random.choice(pos_by_rm_dict['rm'+str(ag1_rm)+'pos']))
    item0_pos = random.choice(pos_by_rm_dict['rm'+str(it0_rm)+'pos'])
    init_positions['item0'].append(item0_pos)
    item1_pos = list(set(pos_by_rm_dict['rm'+str(it1_rm)+'pos'])-set([item0_pos]))
    init_positions['item1'].append(random.choice(item1_pos))




filename = 'independent.txt'
text_file = open(filename, "w")
text_file.write(" ".join(map(str,envs)))
print(" ".join(map(str,envs)))
for param_dict in [alphas, sizes, strengths, angles, goals0, goals1, goals2, goals3, init_positions]:
    for key in sorted(param_dict.keys()):
        text_file.write('\n'+" ".join(map(str,param_dict[key])))
        print(" ".join(map(str,param_dict[key])))
text_file.write('\n')
text_file.close()
