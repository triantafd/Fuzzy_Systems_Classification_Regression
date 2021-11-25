% Triantafyllidis Dimitrios
% Regression with TSK models 
% Superconductivity dataset from UCI repository
tic

format compact
clear
clc


%% Make Folder to Save Results (if doesnt exist)
if ~exist('../results/Regression/Final_model', 'dir')
   mkdir('../results/Regression/Final_model')
end


%% Use seed
rng(1)


%% Load data - Split data
data=load('../Datasets/superconduct.csv');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);
Performance=zeros(1,4);


%% Feature Selection (selects the rank of the features)
% Keep only the number of features we want and not all of them
[ranks, weights] = relieff(data(:,1:end-1), data(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 15 features and radii 0.3 - Substractive Clustering\n');

f = 15;
radii = 0.3;

training_data_x = trnData(:,ranks(1:f));
training_data_y = trnData(:,end);

validation_data_x = chkData(:,ranks(1:f));
validation_data_y = chkData(:,end);

test_data_x = tstData(:,ranks(1:f));
test_data_y = tstData(:,end);

%% TRAIN TSK MODEL MODEL WITH 15 FEATURES AND 4 RULES

% Generate the FIS As input data give only the most important features
fprintf('\n *** Generating the FIS\n');
% genfis2 is used for substractive clustering
init_fis = genfis2(training_data_x, training_data_y, radii);
rules = length(init_fis.rule);

% Tune the fis
fprintf('\n *** Tuning the FIS\n');

% Set some options, set the validation data to avoid overfitting  
% train model
anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
[trn_fis, trainError, stepSize, valFis, valError] = anfis([training_data_x training_data_y], anfis_opt);


%% Plots of MFs before and after training

% Plots of MFs before training
figure;
%for i = 1 : f
 %   [x, mf] = plotmf(init_fis, 'input', i);
  %  plot(x,mf);
   % hold on;
%end
plotMFs(init_fis, 6)
suptitle('TSK model : Membership functions functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, '../results/Regression/Final_model/MFs_before_training.png');

% Plots of MFs after training
figure;
%for i = 1 : f
 %   [x, mf] = plotmf(valFis, 'input', i);
  %  plot(x,mf);
   % hold on;
%end
plotMFs(valFis, 6)
suptitle('Final TSK model : Membership functions functions after training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, '../results/Regression/Final_model/MFs_after_training.png');

%% Validation - Training learning curve
figure;
plot(1:length(trainError), trainError, 1:length(valError), valError);
legend('Training Error','Validation Error');
title('Learning Curve');
saveas(gcf, '../results/Regression/Final_model/learningcurves.png')


%% Evaluate the valFis on the test data
fprintf('\n *** Evaluating the FIS\n');
Y = evalfis(test_data_x, valFis);

%% Plot errors for test Data
error = Y - test_data_y;

figure;
plot(error);
title('Prediction Errors');
saveas(gcf, '../results/Regression/Final_model/Prediction_errors.png')


%% Reference vs real outputs
figure;
plot(1:length(test_data_y), test_data_y, '*r', 1:length(test_data_y), Y, '.b');
title('Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, '../results/Regression/Final_model/output.png')


%%  Calculate metrics for test Data 

% Evaluation function R2 
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% metric
R2=Rsq(Y,test_data_y);
MSE=mse(Y,test_data_y);
RMSE=sqrt(MSE);
NMSE=(sum((test_data_y - Y) .^ 2) / length(Y)) / var(test_data_y);
NDEI=sqrt(NMSE);

Performance(1,:)=[R2;RMSE;NMSE;NDEI];

% Results Table
varnames={'Rsquared','RMSE','NMSE','NDEI'};
Performance=array2table(Performance,'VariableNames',varnames);
fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', MSE, RMSE, R2, NMSE, NDEI)


toc

%% MSE = 195.558848 RMSE = 13.984236 R^2 = 0.831546 NMSE = 0.168414 NDEI = 0.410383
%% Elapsed time is 873.612789 seconds.