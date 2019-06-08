function [RFF] = randomFeature(X_train, dims, type, para)

type = 'Gaussian';

[n,d] = size(X_train);

omega = normrnd(0,para,[d,dims]);
randn('seed',1002312);
omega = randn(d,dims)*sqrt(2*para);

RFF = X_train*omega;

RFF = sqrt(1/dims).*[cos(RFF),sin(RFF)];





