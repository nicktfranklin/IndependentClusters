# IndependentClustering_Demo

This is a demonstration of a reinforcement learning model that
generalizes reward functions and transition functions independently. It is currently a work in progress.


Files:
* setup.py: script that compiles the cython backend for the models. This needs to be run before the demonstration will
run. To compile, run the code:
 `python setup.py build_ext --inplace`
* Demonstration.ipynb: The jupyter notebooks shows a basic comparison in the performance of the models.
* Gridworld.py: Defines the task environments
* Agents.py: Defines the reinforcement learning agents. Calls cython backend.