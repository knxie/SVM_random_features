# SVM_random_features

For RBF kernel:
$$
      k(x_i,x_j) = e^{-s{|x_i-x_j|^2_2}}
$$

SVMcgForClass.m tries to find the best hyperparameters c and g(s) with k-fold cross-validations.
Note that the inputs **cmin,cmax,gmin,gmax** actually are the grid bounds, thus they ought to have negative parts.
Default: -8,8,-8,8 and the step is 0.8.

randomFeature.m tries to compute the random Fourier features of a rbf kernel.
