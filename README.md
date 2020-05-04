# siso_predictor
A predictor of future behavior of a targeted criterion based on input/output profiles
## Possible use
- Forcasting future evolution of an uncontrolled variable from its past 
- Design a Data-Driven Nonlinear Model Predicive Control
## Contents
- The siso_predictor module ```siso_predictor.py```
- The ```main.py``` file  
- A utility ```module generate_data.py``` that is used to generate the data for the test of the module
## Example of use
```
sol = learn_model(
    y=y,
    U=U,
    ydef=ydef,
    N=100,
    n_clusters=3,
    nJump=1,
    max_leaf_nodes=1200,
    test_size=0.33,
    validation_mode='all',
    plot=True
    )
```
