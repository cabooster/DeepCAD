clc;clear;
%% calibration
load('./splited/T.mat')
layers = loadtiff('./splited/layers/2_cut_border.tif');
[xs, ys, ~] = size(layers);

rlt = uint16(zeros([xs, ys, 4]));
for i = 1:4
    img = layers(:,:,i);
    rlt(:,:,i) = imwarp(img, affine2d(T(:,:,i)),'OutputView', imref2d(size(layers)));
end
saveastiff(rlt, './splited/layers/4_border_calib.tif')
