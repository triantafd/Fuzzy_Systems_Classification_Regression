%% Split - Preprocess Data

function [trnData,chkData,tstData] = stratified_split_scale(data,preproc)

    %idx=randperm(length(data));
    c1 = cvpartition(data(:, end),'Holdout',0.4) ;
    trnIdx = c1.training; %logical array 
    val_testIdx = c1.test; %logical array
    val_test_data = data(val_testIdx, 1:end);
    
    c2 = cvpartition(val_test_data(:, end),'Holdout',0.5) ;
    valIdx = c2.training; %logical array 
    tstIdx = c2.test; %logical array 
    
    trnX=data(trnIdx,1:end-1);   
    valX=val_test_data(valIdx,1:end-1);
    tstX=val_test_data(tstIdx,1:end-1);
    
         
    switch preproc
        case 1                      %Normalization to unit hypercube
            xmin=min(trnX,[],1);
            xmax=max(trnX,[],1);
            trnX=(trnX-repmat(xmin,[length(trnX) 1]))./(repmat(xmax,[length(trnX) 1])-repmat(xmin,[length(trnX) 1]));
            valX=(valX-repmat(xmin,[length(valX) 1]))./(repmat(xmax,[length(valX) 1])-repmat(xmin,[length(valX) 1]));
            tstX=(tstX-repmat(xmin,[length(tstX) 1]))./(repmat(xmax,[length(tstX) 1])-repmat(xmin,[length(tstX) 1]));
        case 2                      %Standardization to zero mean - unit variance
            mu=mean(data,1);
            sig=std(data,1);
            trnX=(trnX-repmat(mu,[length(trnX) 1]))./repmat(sig,[length(trnX) 1]);
            valX=(trnX-repmat(mu,[length(valX) 1]))./repmat(sig,[length(valX) 1]);
            tstX=(trnX-repmat(mu,[length(tstX) 1]))./repmat(sig,[length(tstX) 1]);
        otherwise
            disp('Not appropriate choice.')
    end
    trnData=[trnX data(trnIdx,end)];
    chkData=[valX val_test_data(valIdx,end)];
    tstData=[tstX val_test_data(tstIdx,end)];

end