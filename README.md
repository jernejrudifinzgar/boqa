# boqa
Designing quantum annealing schedules with Bayesian optimization

### Usage:

create a conda env and run. I used python 3.9.13.

pip install -r requirements.txt 

to install all of the required dependencies.

For the julia code I used version 1.7.3 -- the packages that need installed can be found at the top of the respective files.

### Short explanation of the corresponding files & their functionalities:

=================================================================================================================

#### aws_helper.py

This is the function deployed within the hybrid jobs framework on AWS and runs the experiments on QuEra. This includes both the whole BO pipeline,
as well as the individual experiments which read in parameters optimized in classical simulations (rydberg_mis.py) and runs them on QuEra.

=================================================================================================================

### easy-graphs, hard-graphs-13-14

Directories containing the graphs, easy and hard in terms of the hardness parameter HP.

=================================================================================================================

#### mis_demo.ipynb

Basic demonstration of the functionalities of the aws_helper.py used to BO schedules for MIS.

=================================================================================================================

#### adquco.py

Main workhorse for the p-spin model data (the unitary case). (Name is derived from ADiabatic QUantum COmputation.)

=================================================================================================================

#### pspin_demo.ipynb

Demo notebook for BO-quantum annealing and BO-reverse annealing the p-spin model.

=================================================================================================================

#### find_hard_graphs.jl

The julia code used to generate random graphs, sort them in terms of their hardness parameter, and return them.

=================================================================================================================

#### ame.jl

Julia code for simulating the Adiabatic Master Equation (AME) using the HOQST julia library.

