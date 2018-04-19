# Compositional Clustering Demo

This is a demonstration of a reinforcement learning model that
generalizes components of task structure independently, for the forthcoming paper 
*Compositional clutering in task structure learning* (Franklin & Frank, *Plos Comp Bio*, 2018). A preprint of 
accompanying these demonstrations is available on bioRxiv: (https://www.biorxiv.org/content/early/2017/10/02/196923)
  
When an simple, artificial agent, such as a Q-learner, encounters a new
task it is required to learn the properties of the task from scratch
via trial and error. A more efficient approach is to generalize 
skills and goals gained in a previous task to a new one. Previous human
subject research has suggested that people generalize rules, or 
task-sets, from one context to another (see Collins & Frank, Psych 
Review, 2013). This behavior is consistent with generalization models
that utilize a non-parametric Bayesian clustering algorithm that 
treats contexts as belonging to a "set" of contexts that share the 
same structure. While this is useful to explain behavior it is limited 
computationally -- in ecological settings both people and artificial
agents are likely to encounter contexts that share only a partial similarity 
with each other.

A more useful approach is to learn pieces of task structure and
generalize them separately. This is particularly useful for
*goal-directed* behavior, a hallmark of which is the ability to combine experiences to 
generate a novel course of action in an unfamiliar environment.

What task components are useful to generalize separately? One possible
division of components is a division between reusable *skills* and 
frequently encountered *goals*. Broadly speaking, skills reflect 
structure in the outcomes of an agents actions whereas goals reflect 
the desirability of various outcomes. In a reinforcement learning 
setting, skills might be thought as generalized options


This repository contains a simplified demonstration that parallels 
 work in human behavior. The repository, and the documentation, are
 currently a work in progress.

#### Notebooks:
* A demonstration of the model's performance can be found in the 
notebook file `Demonstration for paper.ipynb`

* A demonstration of a meta-agent that uses a reinforcement learning 
process to arbitrate between Joint and Independent clustering can be
found in the notebook file `Demonstration - Meta agent.ipynb`

* An information theoretic analysis detailing under what conditions
 it is useful to cluster can be found in `Information Theoretic Analysis.ipynb`

* A demonstration of a problem where the explorations compound
with each new context can be found in `Rooms Problem.ipynb`

* The code to simulate varying the parameters of the rooms problem
can be found in `Rooms Growth.pynb`


___


### Installation Instructions

This library run on Python 2.7 and unlike most python code, requries
 compilation with Cython before use. This requires a C compiler (gcc), 
 [for which you can find documentation here.](
 http://cython.readthedocs.io/en/latest/src/quickstart/install.html)  
 
 If you have already installed python 2.7, pip and gcc on on your system
 , you can install cython and the other dependencies with 
 pip (if you don't already have them), run:  
 ```pip install -r requirements.txt ```

 To compile the cython code, run:  
 ```python setup.py build_ext --inplace```  
  
### Files:
---
* `model.gridworld.py`: Defines the task environments
* `model.agents.py`: Defines the reinforcement learning agents. Core functions 
    rely on cython
* `model.crp.py`: Backend for Normative analysis
* `model.cython_libary`: core functions optomized for speed with cython
* `model.rooms_problem`, `model.rooms_agents`: special agents/models need for rooms simulation