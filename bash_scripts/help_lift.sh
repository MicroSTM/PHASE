#!/bin/bash

# 0 env_ids
# 1 alpha0
# 2 alpha1
# 3 size_a0
# 4 size_a1
# 5 size_i0
# 6 size_i1
# 7 strength_a0
# 8 strength_a1
# 9 angles_a0
# 10 angles_a1
# 11 goal_pre
# 12 goal_ag_it
# 13 goal_lm_it
# 14 goals_end
# 15 init_pos_a0
# 16 init_pos_a1
# 17 init_pos_i0
# 18 init_pos_i1


n_vids=60
for ((i=0;i<n_vids;++i)); do
  #get params
  args=()
  while IFS= read -r line; do
    l=($line)
    args+=(${l[i]})
  done < bash_scripts/help_lift.txt
  echo "params ${args[@]}"
  #same goals
  g=(${args[11]} ${args[12]} ${args[13]} ${args[14]})

	python main_particle.py \
	--max-episode-length 100 --max-nb-episodes 1 --nb-simulations 1000 --max-rollout-steps 10 \
	--levels 0 0 --cInit 1.25 --cBase 1000 --enable-renderer --num-agents 2 --num-items 2 \
	--full-obs 0 0 --alpha ${args[1]} ${args[2]} --all-directions --save-date --action-space-types 0 0 \
  --sizes ${args[3]} ${args[4]} ${args[5]} ${args[6]} \
	--costs 0 0 --strengths ${args[7]} ${args[8]} \
  --init-agent-angles ${args[9]} ${args[10]} \
  --env-id ${args[0]} --goal1 ${g[@]} --goal2 ${g[@]} \
  --init-positions ${args[15]} ${args[16]} ${args[17]} ${args[18]}
done
