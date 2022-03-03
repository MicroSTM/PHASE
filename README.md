# PHASE: PHysically-grounded Abstract Social Events for Machine Social Perception

Code for procedurally synthesizing PHASE animations, proposed in the paper: [*PHASE: PHysically-grounded Abstract Social Events
for Machine Social Perception*](https://www.tshu.io/PHASE/PHASE.pdf).  

The code was written by the lead authors of the paper, Aviv Netanyahu and Tianmin Shu. For more details of the dataset, please visit our [*project website*](https://www.tshu.io/PHASE).

## Requirements

- Python 3.5
- Numpy >= 1.15.2
- Scipy == 1.1.0
- [pybox2d](https://github.com/pybox2d/pybox2d)
- pygame 
- OpenCV >= 3.4.2
- anytree
- matplotlib
- shapely
- tqdm
- pdb


## Instruction

First, run the following commands to sample parameters used for the procedural generation:

 ```
  cd bash_scripts
  bash reset_params.sh
  ```

This will create parameters for several types of animations. Then you can run the script for each type to generate animations for that type. For instance, you may generate collaboration animations by running the following script:

 ```
  cd ..
  bash bash_scripts/collab.sh
  ```

In the script for a specific type of animations, you can specify the number of videos you want to generate for that type through `n_vids`.

Note that not all sampled parameters can lead to the desired interaction. Here we are using heuristics to sample parameters that are more likely to lead to a certain type of animation. To create our original PHASE dataset, we further filtered out videos that were not reflecting the desired type manually. We intend to create an automatic video selection in the future.

## Cite
If you use this code in your research, please consider citing the following paper.

```
@inproceedings{netanyahu2021phase,
  title={{PHASE}: Physically-grounded abstract social events for machine social perception},
  author={Netanyahu, Aviv and Shu, Tianmin and Katz, Boris and Barbu, Andrei and Tenenbaum, Joshua B},
  booktitle={35th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021}
}
```