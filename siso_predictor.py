import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class GenericResults(object):
    # class that contains any kind of result in dedicated fields
    pass


def feature_gen(V=None, order=2):

    # V is a window registering values of some variable
    # each line gathers successive values of this variable.
    # Generates the regressor used in the identification of the model
    # This is precisely the mean values of successive derivatives.

    if len(V.shape) == 1:
        v = V.reshape(1, -1)
    else:
        v = V[:]
    Xfeat = np.zeros((v.shape[0], order+1))
    Xfeat[:, 0] = np.mean(v, axis=1)
    for i in range(order):
        v = np.diff(v, axis=1)
        Xfeat[:, i+1] = np.mean(v, axis=1)

    return Xfeat


def construct_regressor(y=None, U=None, N=None, nJump=None):

    # Extract the past and future values of one sensor together with the
    # past and future values of the input.
    # This implementation is valid only for scalar output and scalar input.
    # (To be generalized in case of success.)
    # although technically not needed, the logic of the program requires that 
    # the sampling be uniform in time. 

    R = GenericResults()

    ind = [i for i in range(len(y)) if i*nJump + 2 * N < len(y)]
    R.Ypast = np.asarray([y[i * nJump:i * nJump + N] for i in ind])
    R.Upast = np.asarray([U[i * nJump:i * nJump + N] for i in ind])
    R.Yplus = np.asarray([y[i * nJump + N:i * nJump + 2 * N] for i in ind])
    R.Uplus = np.asarray([U[i * nJump + N:i * nJump + 2 * N] for i in ind])

    return R

def compute_classfier(R=None, n_clusters=1):

    # Compute blind classification from the learning data
    # R is a typical returned value from the construct_regressor function above.

    Xcl = np.hstack([
        feature_gen(R.Ypast),
        feature_gen(R.Upast),
        feature_gen(R.Uplus)
    ])

    cl = KMeans(n_clusters=n_clusters).fit(Xcl)
    ycl = cl.predict(Xcl)

    return ycl, cl

def learn_model(
    y=None, 
    U=None, 
    ydef=None,
    N=None, 
    nJump=None,
    n_clusters=None,
    max_leaf_nodes=800,
    n_estimators=20,
    nPCA=None,
    test_size=0.33,
    validation_mode='all',
    plot=False
    ):

    # learn a model that is cluster-piece-wise defined.

    R = construct_regressor(y, U, N, nJump)

    ycl, cl = compute_classfier(R=R, n_clusters=n_clusters)

    reg = [RandomForestRegressor(
        max_leaf_nodes=max_leaf_nodes,
        n_estimators=n_estimators)
        for _ in range(n_clusters)
    ]

    data = [GenericResults() for _ in range(n_clusters)]

    XU = np.hstack([feature_gen(R.Ypast),
                    feature_gen(R.Upast),
                    feature_gen(R.Uplus)])

    for icl in range(n_clusters):
        err = []
        ind = [i for i in range(XU.shape[0]) if ycl[i] == icl]
        if nPCA == None:
            Xw = XU[:]
        else:
            Xw = PCA(n_components=nPCA).fit_transform(XU)

        X_all = Xw[ind, :]        
        y_all = np.asarray(
            [
                ydef(
                    ypast=R.Ypast[ind[i], :],
                    upast=R.Upast[ind[i], :],
                    yplus=R.Yplus[ind[i], :], 
                    uplus=R.Uplus[ind[i], :]
                    )
                for i in range(len(ind))]
            )


        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X_all, y_all, test_size=test_size, random_state=42)

        if validation_mode == 'all':

            X_validation = X_all
            y_validation = y_all

        elif validation_mode == 'test':

            X_validation = Xtest
            y_validation = ytest

        else:

            X_validation = Xtrain
            y_validation = ytrain

        y_hat = reg[icl].fit(Xtrain, ytrain).predict(X_validation)
        err = abs(y_validation - y_hat).mean()
        data[icl].yhat = y_hat
        data[icl].err = err
        data[icl].y = y_validation

    sol = GenericResults()
    sol.n_clusters = n_clusters
    sol.R = R
    sol.ycl = ycl
    sol.N = N
    sol.nJump = nJump
    sol.popSize = [np.sum(ycl == j) for j in range(n_clusters)]
    sol.cl = cl
    sol.data = data
    sol.reg = reg
    sol.dim = XU.shape[1]

    if plot:

        cols = ['blue', 'red', 'green', 'magenta',
                'cyan', 'yellow', 'black', 'brown']*10
        legend_list = ['cluster '+str(i+1) 
                        for i in range(sol.n_clusters)]

        for icl in range(sol.n_clusters):      

            yreg = sol.data[icl].y
            yhat = sol.data[icl].yhat
            plt.scatter(yreg, yhat, color=cols[icl], marker='o', alpha=0.4)
            if icl==0:
                vmin = min(yreg.min(), yhat.min())
                vmax = max(yreg.max(), yhat.max())
            else:
                vmin = min(vmin, min(yreg.min(), yhat.min()))
                vmax = max(vmax, max(yreg.max(), yhat.max()))

        plt.grid(True)
        plt.xlim([vmin, vmax])
        plt.ylim([vmin, vmax])
        plt.xlabel('True value')
        plt.ylabel('predicted value')
        plt.title(f"Fitting results with {sol.n_clusters} clusters")
        plt.legend(legend_list)
        plt.plot([vmin, vmax], [vmin, vmax], 'k', linewidth=3)
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    print("-----")
    print(
        f'''Mean abs(Error) on all cluster= {np.array([sol.data[icl].err 
        for icl in range(sol.n_clusters)]).mean():.6f}'''
    )
    print("-----")
    for icl in range(sol.n_clusters):
        print(f"abs(Error) - cluster {str(icl+1)} = {sol.data[icl].err:.6f}")
    print("-----")
        

    return sol


def predict(Ypast=None, Upast=None, Uplus=None, sol=None):

    # function that predicts the average of the Yplus signal using the learned model sol, given in the argument list.

    XU = np.hstack([
        feature_gen(Ypast),
        feature_gen(Upast),
        feature_gen(Uplus)
    ])

    ycl = sol.cl.predict(XU)
    val = np.zeros(XU.shape[0])
    for i in range(len(val)):
        val[i] = sol.reg[ycl[i]].predict(XU[i, :].reshape(1, -1))

    return val
