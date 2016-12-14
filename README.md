# IndependentClustering_Demo

This is a demonstration of a reinforcement learning model that
generalizes reward functions and transition functions independently.  
  
It is currently a work in progress.


### Installation
--- 
This code requires compilation before use. To install cython and the 
other dependencies with pip (if you don't already have them), run:  
```pip install -r requirements.txt ```

To compile the cython code, run:  
```python setup.py build_ext --inplace```  

###Files:
---
* Demonstration.ipynb: The jupyter notebooks shows a basic comparison in
 the performance of the models.
* Gridworld.py: Defines the task environments
* Agents.py: Defines the reinforcement learning agents. Core functions 
    rely on cython