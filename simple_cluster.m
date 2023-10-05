clear;clc;

trainPath = '../savedata/05/';

for iter = 1:6
    tic
    disp(['hmdb_iter1_fl', num2str(iter)]);
    D = [];
    C = [];
    load([trainPath, 'hmdb_iter1_fl', num2str(iter), '_1.mat']);
    [D,C] = sort(testFeat,2,'descend');
%     [D,C] = max(testFeat,[],2);
    clear testFeat;
    D = D(:,1:2);
    C = C(:,1);
    toc
    tic
    load([trainPath, 'hmdb_iter1_fl', num2str(iter), '_2.mat']);
    [Dd,Cc] = sort(testFeat,2,'descend');
    clear testFeat;
    Dd = Dd(:,1:2);
    Cc = Cc(:,1);
    C = [C;Cc];
    D = [D;Dd];
    toc
    save([trainPath, 'hmdb51_c1_c256', num2str(iter), '.mat'], 'C', 'D');
end


