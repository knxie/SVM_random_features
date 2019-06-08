clear all;
close all;
clc;
% n = 10000;
% d = 10;
% Xtrain = randn(d, n);
% Xtest = randn(d, n/10);
% 
% ytrain = sqrt(sum(Xtrain.*Xtrain,1))>sqrt(d);
% ytrain = ytrain'*2-1;
% ytest = sqrt(sum(Xtest.*Xtest,1))>sqrt(d);
% ytest = ytest'*2-1;


file = '/home/knight/桌面/deep_rff/UCI/';
dataSet = 'monks_2.mat'

load([file dataSet])

% whiten data

trainAcc = [];
testAcc = [];


for i=1:1
    X_min = repmat(min( X ), [size(X, 1),1]);
    X_max = repmat(max( X ), [size(X, 1),1]);
    X = (X - X_min)./(X_max - X_min);
    
    if flagtest == 0
        rate = 1/2;
        Training_num = round(length(Y)*rate);
        [temp, index] = sort(rand( length(Y), 1));
        X_train = X( index( end - Training_num+1 : end), : );
        Y_train = Y( index( end - Training_num+1: end));
        X_test = X( index( 1 : end - Training_num), : );
        Y_test = Y( index( 1 : end - Training_num));
        
%         X_train = X(round(Training_num*0.2)+1:end,:)
%         X_test = X(1:round(Training_num*0.2),:);
%         Y_train = Y(round(Training_num*0.2)+1:end)
%         Y_test = Y(1:round(Training_num*0.2));
    else
        X_train = X; Y_train = Y;
        X_min = repmat(min( X_test ), [size(X_test, 1),1]);
        X_max = repmat(max( X_test ), [size(X_test, 1),1]);
        X_test = (X_test - X_min)./(X_max - X_min);
    end
    X_nor = X_train;
    X_test_nor = X_test;
   % model_rbf=svmtrain(Y_train, X_train, '-s 0 -t 2 -c 1 -g 0.1 ')
  %  [bestacc,bestc,bestg] = SVMcgForClass(Y_train,X_train,-5,10,-5,10,5,1,1,0.1)
%
    para = fprintf('-s 0 -t 2 -c %.2f -g %.2f',8, 0.25);
    model_rbf=svmtrain(Y_train, X_train, para);
    [~,acc,~] = svmpredict(Y_test,X_test, model_rbf);
   
    
    [nTrain,d] = size(X_train);
    [nTest,d2] = size(X_test);
    
    k_type = 'TL1';
    k_par = 0.4*d;
    
    tl1Ktrain = create_kernel2(X_train,X_train,k_type,k_par);
    tl1Ktest = create_kernel2(X_train,X_test,k_type,k_par);
    tl1Para = fprintf('-t 4 -c %.2f',10);
    modelTl1 = svmtrain(Y_train,[(1:nTrain)',tl1Ktrain],tl1Para);
    [~,accTl1,~] = svmpredict(Y_train,[(1:nTrain)',tl1Ktrain], modelTl1);
    [predict,accTl2,~] = svmpredict(Y_test,[(1:nTest)',tl1Ktest], modelTl1);
    
     
     
    testAcc = [testAcc,acc(1)];
 end


fprintf('---testAcc:%.2f+-%.2f\n',mean(testAcc),std(testAcc))