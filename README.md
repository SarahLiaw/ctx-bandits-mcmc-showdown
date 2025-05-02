For linear todo (April 29th):
0. Changed how beta is initialized and write stochastic gradinet and stochastic variance reduction code.
1. In run, first save the .pt file and run it on all the algorithms.
2. Repeat 1 10 times over 10 initiatlizations and saving of the .pt file. Does this mean changing the seed or sth?
3. In linear_results, Produce a plot that gives the mean regret and the variance plotted over the 10 runs as like a blurred confidence region among the mean regret, save the final values of this cum regret and variance for each of this algorithm over the 10 runs.

- This is for changing arms. 

TODO: Implement beta, and fix stochastic graidnet (suppose it's done)
Do d= 20
d = 100
### Synthetic data

To run the bandit algorithm:

python3 run.py --config_path config/toy/mala.json --exp_idx 0

This command reads the configuration you want to evaluate (check in folder config/toy), generates you synthetic data. And then It runs the experiment and save the result in the folder result. The attribute idx_exp allows to run your experiment many times on several devices using a Slurm array without overwriting your result file. 


### Yahoo ! Front Page

Firstly, we have to load the Yahoo dataset: "R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)". The dataset can be found in https://webscope.sandbox.yahoo.com/catalog.php?datatype=r.

Then, to run the yahoo experiment:


python3 run.py --config_path config/yahoo/fgmala_yahoo.json --exp_idx 0


This command is similar to the one for the toy example.

### Logistic

This experiment is based on the folder Logistic. Firstly, you have to generate the synthetic data. Please, go on the folder Logistic and run

python3 data_generator.py --config configs/gauss_bandit_baseline.yaml


Then, run bandit algorithm using the command

python3 run_simulation.py --config_path configs/logistic/lmc.yaml --repeat 10 --log