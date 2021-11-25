% Triantafyllidis Dimitrios
% Regression with TSK models 
% Habermans Survival dataset from UCI repository
tic

format compact
clear 
clc


%% Make Folder to Save Results (if doesnt exist)
if ~exist('../results/Classification/Cls_2', 'dir')
   mkdir('../results/Classification/Cls_2')
end

%% Use seed
rng(1)

%% Load data
data=load('../Datasets/haberman.data');


%%Balance Dataset (if needed)
%{
class2=data(find(data(:,4)==2),:);
class1=data(find(data(:,4)==1),:);

class1=class1(1:length(class2),:);
data=[class1',class2'];
data=data';
%}


%% init arrays
Acc=zeros(2,1);
numOfRules=zeros(2,1);
Perf=zeros(1,6);


%% split the data to 60 training,20 val and 20 test.
preproc=1;
[trnData,valData,tstData]=stratified_split_scale(data,preproc);


%% ANFIS - Scatter Partition

options = genfisOptions('SubtractiveClustering');
options.ClusterInfluenceRange = 0.9;
% or do that model=genfis2(trnData(:,1:end-1),trnData(:,end),0.7);
model = genfis(trnData(:,1:end-1), trnData(:,end), options); 

%get the num of rules
numOfRules=length(model.rule());
%numOfRules=length(model.Rules());
[trnFis,trnError,~,valFis,valError]=anfis(trnData,model,[100 0 0.01 0.9 1.1],[],valData);

for j = 1:length(model.output(1).mf)
    model.output(1).mf(j).type = 'constant';
end

%% Plots of MFs before and after training

% Plot some input membership functions
figure;
plotMFs(model,size(trnData,2)-1);
suptitle('TSK model : some membership functions before training');
saveas(gcf,'../results/Classification/Cls_2/MFs_before_training.png')

% Plot the input membership functions after training
figure;
plotMFs(valFis,size(trnData,2)-1);
suptitle('TSK model : some membership functions after training');
saveas(gcf,'../results/Classification/Cls_2/MFs_after_training.png')


%% Plot Learning curve
figure;
plot(1:length(trnError), trnError, 1:length(valError), valError);
title('TSK model : Learning Curve');
xlabel('Iterations');
ylabel('Error');
legend('Training Set', 'Validation Set');
saveas(gcf,'../results/Classification/Cls_2/Learning_Curve.png')

%% Evaluate the valFis model on test data
Y=evalfis(tstData(:,1:end-1),valFis);
Y=round(Y);

for i=1:length(Y)
    if Y(i) > 2
        Y(i) = 2;
    end
    if Y(i) < 1
        Y(i) = 1;
    end
end

diff=tstData(:,end)-Y;
Acc=(length(diff)-nnz(diff))/length(Y)*100;


%% Error Matrix
error_matrix = confusionmat(tstData(:,end), Y);
pa = zeros(1, 2);
ua = zeros(1, 2);

%% confusion matrix
%figure;
%cm = confusionchart(tstData(:,end),Y);


%% Overall accuracy - OA
overall_acc = 0;
N = length(tstData);
for i = 1 : 2
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N;


%% Producer’s accuracy (PA) – User’s accuracy (UA)
for i = 1 : 2
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end


%% k
p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
pe = p1 + p2;
k = (overall_acc - pe) / (1 - pe);

fprintf('OA = %f K = %f \n', overall_acc, k);
fprintf('PA = %f \n', pa);
fprintf('UA = %f \n', ua);
toc

%{ 
OA = 0.737705 K = 0.229068 
PA = 0.784314 
PA = 0.500000 
UA = 0.888889 
UA = 0.312500 
%}
%% Elapsed time is 2.063375 seconds.