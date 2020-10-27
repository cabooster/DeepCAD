% get layer
clc;clear;close all
%% load split data
A_split = cell(4);
for i = 1:4
    load(['./splited/' num2str(i) '/resP1.mat']);
    A_split{i} = reshape(full(A), 256,256, size(full(A), 2));
end

%% load 4p
load('./4p/1/resV1_trs.mat')
A_4p = reshape(full(A), 256,256, size(full(A), 2));
layer_recorder = zeros(size(A_4p,3),1);
for i = 1:size(A_4p, 3)
    comp = A_4p(:,:,i);
    mask = (comp~=0);
    for j =1:4
        A_basic = A_split{j};
        for jj =1:size(A_basic, 3)
            mask_basic = (A_basic(:,:,jj)~=0);
            if sum(sum((int8(mask)-int8(mask_basic))~=0)) ==0
                layer_recorder(i,1) = j;
                fprintf('No. %d, layer %d\n',i,j)
            end
            
        end
        
    end
end

save('./layer_recorder.mat', 'layer_recorder')