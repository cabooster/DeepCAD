clear;clc;close all
%% load 4p
path = './4p/1/';
load([path 'resV1_trs.mat']);
C_4p = C;
A_comp = full(A);
load([path 'snrV1_xyt_trs.mat'])
snr_4p = snrs;
%% load ref
path = './ref/1/';
load([path 'resV1_trs.mat']);
C_ref = C;
load([path 'snrV1_xyt_trs.mat'])
snr_ref = snrs;

%% plot SNR
load('layer_recorder.mat')
% figure;plot(snr_4p);hold on;plot(snr_ref)
figure;
snr_3_4_mask = int8(layer_recorder ==3) + int8(layer_recorder ==4);
snr_3_4_mask = snr_3_4_mask~=0;
snr_4p = snr_4p(snr_3_4_mask);
snr_ref = snr_ref(snr_3_4_mask);
[snr_4p_sort,idx_4p] = sort(snr_4p, 'ascend');
[snr_ref_sort, idx_ref] = sort(snr_ref,'ascend');
plot(snr_4p_sort(95:end-1), 'r-', 'Color', [62 133 198]/255, 'Linewidth',2);
hold on
plot(snr_ref_sort(95:end-1), 'r-', 'Color', [222 30 39]/255, 'Linewidth',2);
set(gca, 'LineWidth', 2, 'FontSize',13)
axis([0 330 0 30])


%% plot some traces
% idx = idx_4p((end-130):end-121);
% idx = [15;55;189;124;7;18;79;120;410;12];
idx = [79;124;7;18;12;15;317;114;62;285];
layer = [];
% idx = sort(idx, 'ascend');
trace_4p = C_4p(idx,:);
trace_4p = line_norm(trace_4p);
bias = bias_mat(trace_4p,1);
figure;
tra = (trace_4p+bias);
subplot(1,2,1);
for i = 1:10
    if i >5
        plot(tra(i,:),'Color', [146,84,255]/255)
    else
        plot(tra(i,:),'Color', [0 197 205]/255)
    end
    hold on
end
axis([-15 750 -0.5 10.5])
xticklabels({'0','40','80','120'})
yticklabels({'1','2','3','4', '5','6','7','8','9','10'})
set(gca, 'LineWidth', 1.5, 'FontSize',22)
hold off

trace_ref = C_ref(idx,:);
trace_ref = line_norm(trace_ref);
bias = bias_mat(trace_ref,1);
tra = (trace_ref+bias);
subplot(1,2,2);
for i = 1:10
    if i >5
        plot(tra(i,:),'Color', [146,84,255]/255)
    else
        plot(tra(i,:),'Color', [0 197 205]/255)
    end
    hold on
end
hold off
axis([-15 750 -0.5 10.5])
xticklabels({'0','40','80','120'})
yticklabels({'1','2','3','4', '5','6','7','8','9','10'})
set(gca, 'LineWidth', 1.5, 'FontSize',22)

buf = A_comp(:,idx);
A_sel = uint16(zeros(256,256,size(buf,2)));
for i = 1:size(buf,2)
    tmp = reshape(buf(:,i),256,256);
    tmp = tmp-min(tmp(:));
    tmp = tmp/max(tmp(:));
    tmp = uint16(tmp * 65535);
    A_sel(:,:,i) = tmp;
end
saveastiff(A_sel, 'selected_components.tif')


% %% plot split
% cm = [255 47 39;
%          143 255 89;
%          70 255 255;
%          146 84 255];
% D = 256*3;
% r_border = zeros(D,D);g_border = zeros(D,D);b_border = zeros(D,D);
% r_inner = zeros(D,D);g_inner = zeros(D,D);b_inner = zeros(D,D);
% for j = 1:4
%     path = ['./splited/' num2str(j) '/'];
%     load([path 'resP1.mat']);
%     % plot spatial components
%     A = full(A);
%     A_full = zeros(256,256,size(A,2));
%     A_border = zeros(D,D,size(A,2));
%     A_inner = zeros(D,D,size(A,2));
%     n = size(A, 2);
%     for i = 1:size(A,2)
%         this_one = reshape(A(:,i),256,256);
%         this_one = this_one / max(this_one(:));
%         [border,inner] = getBorder(imresize(this_one, [D,D]));
%         A_border(:,:,i) = border;
%         A_inner(:,:,i) = inner;
%         A_full(:,:,i) = this_one;
%     end
%     % get components
%     A_proj = sum(A_full, 3);
%     A_proj = A_proj/max(A_proj(:))*255;
%     A_base = imresize(A_proj , [D,D]);
%     % get all border
%     A_border_proj = sum(A_border, 3);
%     A_border_proj = imbinarize(A_border_proj, 0.001);
%     % get all inner
%     A_inner_proj = sum(A_inner, 3);
%     A_inner_proj = imbinarize(A_inner_proj, 0.001);
%
%     figure;
%     subplot(1,3,1),imshow(A_base, []);title('All spatial components')
%     subplot(1,3,2),imshow(A_border_proj, []);title('border')
%     subplot(1,3,3),imshow(A_inner_proj, []);title('inner')
%
%     imwrite(uint8(A_proj), ['./splited/layers/components_' num2str(j) '_' num2str(n) '.tif'])
%     imwrite(uint8(A_border_proj*65535), ['./splited/layers/border_' num2str(j) '_' num2str(n) '.tif'])
%     imwrite(uint8(A_inner_proj*65535), ['./splited/layers/inner_' num2str(j) '_' num2str(n) '.tif'])
% end


