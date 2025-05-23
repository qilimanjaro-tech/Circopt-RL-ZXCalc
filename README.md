# Circopt-RL-ZXCalc
This repository contains the code used to produce the results of the publication [Reinforcement Learning Based Quantum Circuit Optimization via ZX-Calculus](https://arxiv.org/abs/2312.11597). 

## Results and data verification
* The data to create the figures of the paper are inside the folder ```results/5x60_non_clifford```. The ```json``` data files where generate with ```agent_test.py```. The code used to generate the figures is in the file ```results.ipynb```.
The code used to create Table 2 from the paper can be found in the file ```benchmark.py```.

* The optimized circuits can be found in the ```rl-zx/results/circuits``` path:
    * **Original**: Contains the original circuits.
    * **gflow-cflow-opt**: Contains the optimized circuits using the combination of [Staudacher et al.](https://arxiv.org/abs/2311.08881) and [Holker](https://arxiv.org/abs/2312.02793)
    * **NRSCM**: Contains the optimized circuits using the algorithm from [Nam et al.](https://arxiv.org/abs/1710.07345).
    * **rl-zx-opt**: Contains the optimized circuits using the RL agent presented in this paper. 

* In ```rl-zx/results/data```are the ```json```files used to generate Figure 11.

## Code

* The agent architecture can be found in the file ```rl_agent```. 
* The Proximal Policy Optimization (PPO) algorithm used to train our agent is in the file ```ppo.py```.
* The environment can be found in ```rl-zx/gym-zx/envs``` with the name ```zx_env.py```
* To retrain the agent simply run ```python ppo.py```. Note that to ensure reproducibility, the hyperparameters of the agent need to be the ones from Table 1 in our [paper](https://arxiv.org/abs/2312.11597). Here is an example of how you can tune them:
```
python ppo.py --exp-name agent_train --learning-rate 1e-3 --total-timesteps 10000000 --num-envs 8 --anneal-lr True --update-epochs 8 --max-grad-norm 0.5 --num-steps 1024 --num-minibatches 16 --vf-coef 0.5 --ent-coef 0.01 --clip-vloss True --clip-coef 0.1 --gamma 0.995 --gae-lambda 0.95
```
## Installation 
To train the agent or verify the results, please create an environment with python = 3.10.
```
conda create -n gym-zx python=3.10
conda activate gym-zx
```
And install the corresponding packages
```
    python installation.py
```

## Citation
```ruby
@misc{riu2024reinforcement,
      title={Reinforcement Learning Based Quantum Circuit Optimization via ZX-Calculus}, 
      author={Jordi Riu and Jan Nogué and Gerard Vilaplana and Artur Garcia-Saez and Marta P. Estarellas},
      year={2024},
      eprint={2312.11597},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
