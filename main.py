import numpy as np
from generate_data import generate_random_excitation, simulate_OL
from siso_predictor import learn_model

# Generate the data set 
t, U = generate_random_excitation()
x0 = np.array([0.2, 0.1, 0.1])
X = simulate_OL(x0, U)
y = X[:, 1]

# Define the output to be identified as a function 
# of the past input/output measurement 

def ydef(ypast=None, upast=None, yplus=None, uplus=None):

    return yplus.mean()-ypast.mean()

# Learn the model using the siso_predictor module. 
# if plot=True, the result will be plotted. 

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

# Test the designed predictor

from siso_predictor import construct_regressor, predict

# Make sure that the same window length is used as the one 
# used to build the predictor. 
R = construct_regressor(N=sol.N, y=y, U=U, nJump=1)

# Extract a slice of the simulated data to call the predictor with
ind = range(20, 500)
Ypast = R.Ypast[ind, :]
Upast = R.Upast[ind, :]
Uplus = R.Uplus[ind, :]
Yplus = R.Yplus[ind, :]

z = np.array([ydef(
    ypast=Ypast[i, :], 
    upast=Upast[i, :], 
    yplus=Yplus[i, :], 
    uplus=Uplus[i, :])
        for i in range(len(ind))])

# Call the predictor 
z_predicted = predict(Ypast, Upast, Uplus, sol)

# produce statistics on the error 
print('Testing the computed predictor')
print(f"Mean_abs_error = {abs(z-z_predicted).mean()}")
