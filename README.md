# IndependentClustering_Demo

This is a demonstration of a reinforcement learning model that
generalizes reward functions and transition functions independently.  
  
It is currently a work in progress.

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
 , you can install To install cython and the other dependencies with 
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