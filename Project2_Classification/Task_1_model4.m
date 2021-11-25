% Triantafyllidis Dimitrios
% Regression with TSK models 
% Habermans Survival dataset from UCI repository
tic

format compact
clear 
clc


%% Make Folder to Save Results (if doesnt exist)
if ~exist('../results/Classification/Cls_4', 'dir')
   mkdir('../results/Classification/Cls_4')
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
[trnData,chkData,tstData]=stratified_split_scale(data,preproc);

%% ANFIS - Scatter Partition

%%Clustering Per Class
radius=0.9;
[c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
[c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
num_rules=size(c1,1)+size(c2,1);

%Build FIS From Scratch
fis=newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in={'in1','in2','in3'};
for i=1:size(trnData,2)-1
    fis=addvar(fis,'input',names_in{i},[0 1]);
end
fis=addvar(fis,'output','out1',[1 2]);

%Add Input Membership Functions
name='clus';
name1='cat1';
name2='cat2';

for i=1:size(trnData,2)-1
    for j=1:size(c1,1)
        fis=addmf(fis,'input',i,strcat(name1,name,int2str(j)),'gaussmf',[sig1(i) c1(j,i)]);
        %fis=addmf(fis,names_in{i},'gaussmf',[sig1(j) c1(j,i)]);
    end
    for j=1:size(c2,1)
        fis=addmf(fis,'input',i,strcat(name2,name,int2str(j)),'gaussmf',[sig2(i) c2(j,i)]);
        %fis=addmf(fis,names_in{i},'gaussmf',[sig2(j) c2(j,i)]);
    end
end

%Add Output Membership Functions
params=[zeros(1,size(c1,1)) ones(1,size(c2,1))];
%params=[ones(1,size(c1,1)) 2*ones(1,size(c2,1))];
for i=1:num_rules
    fis=addmf(fis,'output',1, name,'constant',params(i));
    %fis=addmf(fis,'out1','constant',params(i));
end

%Add FIS Rule Base
%bxcvc = size(trnData,2);
ruleList=zeros(num_rules,size(trnData,2));
for i=1:size(ruleList,1)
    ruleList(i,:)=i;
end

ruleList=[ruleList ones(num_rules,2)];
fis=addrule(fis,ruleList);

%fis=genfis2(trnData(:,1:end-1),trnData(:,end),radius);
    
%get the num of rules
numOfRules(i)=length(fis.rule());
%if matlab matlab>2018 user code below
%numOfRules(i)=length(fis.Rules()); 
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);

%% Plots of MFs before and after training
% Plot input membership functions before training
figure;
plotMFs(fis,size(trnData,2)-1);
suptitle('TSK model : some membership functions before training');
saveas(gcf,'../results/Classification/Cls_4/MFs_before_training.png')

% Plot the input membership functions after training
figure;
plotMFs(valFis,size(trnData,2)-1);
suptitle('TSK model : some membership functions after training');
saveas(gcf,'../results/Classification/Cls_4/MFs_after_training.png')


%% Plot Learning curve
figure;
plot(1:length(trnError), trnError, 1:length(valError), valError);
title('TSK model : Learning Curve');
xlabel('Iterations');
ylabel('Error');
legend('Training Set', 'Validation Set');
saveas(gcf,'../results/Classification/Cls_4/Learning_Curve.png')


%evaluation of model
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
    

%% Producer’s accuracy – User’s accuracy
N = length(tstData);
for i = 1 : 2
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end

%% Overall accuracy
overall_acc = 0;
for i = 1 : 2
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N;

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
OA = 0.721311 K = 0.199228 
PA = 0.780000 
PA = 0.454545 
UA = 0.866667 
UA = 0.312500 
%}
%% Elapsed time is 2.272274 seconds.