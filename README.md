# boqa
Designing quantum annealing schedules with Bayesian optimization

Usage:

create a conda env and run. I used python 3.9.13.

pip install -r requirements.txt 

to install all of the required dependencies.

For the julia code I used version 1.7.3 -- the packages that need installed can be found at the top of the respective files.

Short explanation of the corresponding files & their functionalities:

=================================================================================================================
aws_helper.py

This is the function deployed within the hybrid jobs framework on AWS and runs the experiments on QuEra. This includes both the whole BO pipeline,
as well as the individual experiments which read in parameters optimized in classical simulations (rydberg_mis.py) and runs them on QuEra.

=================================================================================================================

