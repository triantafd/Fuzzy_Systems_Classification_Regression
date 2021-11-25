% Triantafyllidis Dimitrios
% Regression with TSK models 
% Airfoil Self-Noise dataset from UCI repository
tic

format compact
clear
clc

%% Make Folder to Save Results (if doesnt exist)
if ~exist('../results/Regression/Reg_4', 'dir')
   mkdir('../results/Regression/Reg_4')
end

%% Use seed
rng(1)

%% Load data - Split data
data=load('../Datasets/airfoil_self_noise.dat');
preproc=1;
[trnData,valData,tstData]=split_scale(data,preproc);
Performance=zeros(1,4);

%% FIS with grid partition
%genfis1 used for grid partition

Model = genfis1(trnData, 3, 'gbellmf','linear');
[trnFis,trnError,~,valFis,valError]=anfis(trnData,Model,[100 0 0.01 0.9 1.1],[],valData);


%% Plots of MFs before and after training

%~ Plot Membership Functions of Initial Model ~%
figure(1);
plotMFs(Model, size(trnData,2)-1);
suptitle('TSK model : Membership functions before training');
saveas(gcf,'../results/Regression/Reg_4/MFs_before_training.png')

%~ Plot Membership Functions of Trained Model ~%
figure(2);
plotMFs(valFis, size(trnData,2)-1);
suptitle('TSK model : Membership functions after training');
saveas(gcf,'../results/Regression/Reg_4/MFs_after_training.png')


%% Validation - Training learning curve
figure(3);
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation Curve');
saveas(gcf,'../results/Regression/Reg_4/Learning_Curve.png')


%% Evaluate the valFis model on test data
Y=evalfis(tstData(:,1:end-1),valFis);


%% Plot errors for test Data
figure(4);
error = Y - tstData(:,end);
plot(error);
title('Prediction Errors');
saveas(gcf,'../results/Regression/Reg_4/Prediction_errors.png')


%% Calculate metrics for test Data 
% Evaluation function Rsquared
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% metrics
R2=Rsq(Y,tstData(:,end));
RMSE=sqrt(mse(Y,tstData(:,end)));
NMSE=(sum((tstData(:,end) - Y) .^ 2) / length(Y)) / var(tstData(:,end));
NDEI=sqrt(NMSE);

% place matrix into a matrix called Performance (Performanceormance)
Performance(1,:)=[R2;RMSE;NMSE;NDEI];

% Results Table
varnames={'Rsquared','RMSE','NMSE','NDEI'};
Performance=array2table(Performance,'VariableNames',varnames);
fprintf('R^2 = %f RMSE = %f  NMSE = %f NDEI = %f\n', R2, RMSE, NMSE, NDEI)

toc

%% R^2 = 0.793893 RMSE = 3.146663  NMSE = 0.205422 NDEI = 0.453235
%% Elapsed time is 3018.045487 seconds.
