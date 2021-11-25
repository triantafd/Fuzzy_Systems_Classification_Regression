tic

format compact
clear 
clc


%% Make Folder to Save Results (if doesnt exist)
if ~exist('../results/Classification/Final_model', 'dir')
   mkdir('../results/Classification/Final_model')
end

%% Use seed
rng(1)

%% Load data - Split data
data=readtable('../Datasets/epileptic_seizure_data.csv');
%the following line works for newer verrsios of matlab 2018b
%data=removevars(data,{'Var1'});
data.Var1 = [];
data= table2array(data);

preproc=1;
[trnData,chkData,tstData]=stratified_split_scale(data,preproc);


% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(data(:,1:end-1), data(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 10 features and radii 0.3 - Substractive Clustering\n');

f = 10;
radii = 0.3;

training_data_x = trnData(:,ranks(1:f));
training_data_y = trnData(:,end);

validation_data_x = chkData(:,ranks(1:f));
validation_data_y = chkData(:,end);

test_data_x = tstData(:,ranks(1:f));
test_data_y = tstData(:,end);%% TRAIN TSK MODEL

%% MODEL WITH 10 FEATURES AND 4 RULES
% Generate the FIS
fprintf('\n *** Generating the FIS\n');
model = genfis2(training_data_x, training_data_y, radii);
rules = length(model.rule);

% Tune the fis
fprintf('\n *** Tuning the FIS\n');
anfis_opt = anfisOptions('InitialFIS', model, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);
[trnFis,trnError,~,valFis,valError] = anfis([training_data_x training_data_y], anfis_opt);


%% Plots of MFs before and after training
% Plot some input membership functions
figure;
plotMFs(model,size(training_data_x,2)-7);
suptitle('Final TSK model : some membership functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, '../results/Classification/Final_model/some_mf_before_training.png');

% Plot the input membership functions after training
figure;
plotMFs(valFis,size(training_data_x,2)-7);
suptitle('Final TSK model : some membership functions after training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, '../results/Classification/Final_model/mf_after_training.png');

%% Plot Learning curve
figure;
plot(1:length(trnError), trnError, 1:length(valError), valError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
saveas(gcf, '../results/Classification/Final_model/learningcurves.png')


%% Evaluate the fis
fprintf('\n *** Evaluating the FIS\n');
%evaluation of model
Y = evalfis(test_data_x, valFis);
Y=round(Y);

for i=1:length(Y)
    if Y(i) > 5
        Y(i) = 5;
    end
    if Y(i) < 0.5
        Y(i) = 1;
    end
end

diff=tstData(:,end)-Y;
Acc=(length(diff)-nnz(diff))/length(Y)*100;

%% METRICS

%% Error Matrix
error_matrix = confusionmat(tstData(:,end), Y);
pa = zeros(1, 5);
ua = zeros(1, 5);

%% confusion matrix
%figure;
%cm = confusionchart(tstData(:,end),Y);

error = Y - test_data_y;

%% Producer’s accuracy – User’s accuracy
N = length(tstData);
for i = 1 : 5
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end

%% Overall accuracy
overall_acc = 0;
for i = 1 : 5
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N;

%% k
p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
p3 = sum(error_matrix(3, :)) * sum(error_matrix(:, 3)) / N ^ 2;
p4 = sum(error_matrix(4, :)) * sum(error_matrix(:, 4)) / N ^ 2;
p5 = sum(error_matrix(5, :)) * sum(error_matrix(:, 5)) / N ^ 2;
pe = p1 + p2 + p3 + p4 + p5;
k = (overall_acc - pe) / (1 - pe);

% Plot the metrics
figure;
plot(1:length(test_data_y), test_data_y, '*r', 1:length(test_data_y), Y, '.b');
title('Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, '../results/Classification/Final_model/output.png')


fprintf('OA = %f K = %f \n', overall_acc, k);
fprintf('PA = %f \n', pa);
fprintf('UA = %f \n', ua);
toc

%{ 
OA = 0.387391 K = 0.234239 
PA = 0.931624 
PA = 0.300000 
PA = 0.293054 
PA = 0.286494 
PA = 0.066667 
UA = 0.710870 
UA = 0.097826 
UA = 0.669565 
UA = 0.456522 
UA = 0.002174 
%}
%% Elapsed time is 250.856566 seconds..