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
* The environment can be found in ```rl-zx/gym-zx/envs``` with the name ```zx_env.py```

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
