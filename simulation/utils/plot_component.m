% plot all selected components
clc;clear;
load('./4p/A.mat');
load('./4p/spikes.mat')
baseimg = imread('STD_4p.png');
baseimg = uint8(baseimg/max(baseimg(:))*255);
[xs,ys,~]=size(baseimg);
% baseimg = uint8(ones(xs,ys)) *255;  % white background
cm=[255 47 39;
    143 255 89;
    70 255 255;
    146 84 255];

rc = baseimg;
gc = baseimg;
bc = baseimg;

for i = 1:size(A,3)
    comp = A(:,:,i)';
    [border, inner] = getBorder(comp, 0.01);
    
    
    % change color
    rc(border)=cm(1,1);
    gc(border)=cm(1,2);
    bc(border)=cm(1,3);
    
end

%
baseimg_c = uint8(zeros(xs,ys,3));
baseimg_c(:,:,1) = rc; baseimg_c(:,:,2) = gc; baseimg_c(:,:,3) = bc;
figure;imshow(baseimg_c)

%% ref
load('./ref/A.mat');
load('./ref/spikes.mat')
baseimg = imread('STD_ref_local.tif');
baseimg = uint8(baseimg/max(baseimg(:))*255*2);
[xs,ys,~]=size(baseimg);
% baseimg = uint8(ones(xs,ys)) *255;  % white background
cm=[255 47 39;
    143 255 89;
    70 255 255;
    146 84 255];

rc = baseimg_c(:,:,1);
gc = baseimg_c(:,:,2);
bc = baseimg_c(:,:,3);

for i = 1:size(A,3)
    comp = A(:,:,i)';
    [border, inner] = getBorder(comp, 0.1);
    
    
    % change color
    rc(border)=cm(2,1);
    gc(border)=cm(2,2);
    bc(border)=cm(2,3);
    
end

%
baseimg_c = uint8(zeros(xs,ys,3));
baseimg_c(:,:,1) = rc; baseimg_c(:,:,2) = gc; baseimg_c(:,:,3) = bc;
figure;imshow(baseimg_c)