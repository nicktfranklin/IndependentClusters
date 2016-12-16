# IndependentClustering_Demo

This is a demonstration of a reinforcement learning model that
generalizes reward functions and transition functions independently.  
  
The broad purposes of this is to show that it is advantageous for a
reinforcement learner (RL) to generalize information about its goals 
separately from information about the effects of its actions. In RL
 terms, this corresponds to generalizing its reward functions
 separately from transition functions. Here, we use a Bayesian model
 that clusters contexts with a popularity-based generative process.
 Previous human subject research has suggested that people generalize
 between contexts consistent with such an approach (see, Collins & Frank,
 Psych Review, 2013) but here we extend the work to multistep and
 goal-directed domains, and add in basic conditionality by allowing
 independent generalization of reward and transition functions.

This repository contains a simplified demonstration that parallels 
 work in human behavior. The repository, and the documentation, are
 currently a work in progress.

### Demonstration
---
A demonstration of the model's performance can be found in the 
notebook file `Demonstration.ipynb`


### Installation Instructions
--- 
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
  
###Files:
---
* `Demonstration.ipynb`: The jupyter notebooks shows a basic comparison in
 the performance of the models.
* `Gridworld.py`: Defines the task environments
* `Agents.py`: Defines the reinforcement learning agents. Core functions 
    rely on cython