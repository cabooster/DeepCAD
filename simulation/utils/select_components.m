% select conponent
clc;clear;close all;
%% show base images
img_gt = imread('STD_basic.png');
img_4p_ = imread('STD_4p.png'); img_4p=uint8(zeros(442,424,3));img_4p(:,:,1)=img_4p_;img_4p(:,:,2)=img_4p_;img_4p(:,:,3)=img_4p_;
img_ref_ = imread('STD_ref.png');img_ref=uint8(zeros(442,424,3));img_ref(:,:,1)=img_ref_;img_ref(:,:,2)=img_ref_;img_ref(:,:,3)=img_ref_;
figure(1);
subplot(1,3,1);imshow(img_gt,[]);
subplot(1,3,2);imshow(img_4p,[]);
subplot(1,3,3);imshow(img_ref,[]);

%% load CNMF results
load('./1/4p/A.mat'); A_4p = A;
load('./2/ref/A.mat'); A_ref = A;

cn_4p = size(A_4p, 3);
cn_ref = size(A_ref, 3);
bbb = ones(3, 3);
cm = jet(2)*255;

k=0;
for i =1:100000000
    x = input('Press Enter to get next.','s');
    if strcmp(x,'q')
        break
    end
    if strcmp(x, 'u')
        k=k-2;
    end
    
    k=k+1;
    imgToSee_gt=img_gt;
    imgToSee_4p=img_4p;
    imgToSee_ref=img_ref;
    
    % 4p mask
    comp_4p = A_4p(:,:,k)' ;
    mask_4p = imbinarize(comp_4p, 0.01);
    border_4p = imdilate(mask_4p, bbb) - imerode(mask_4p, bbb);
    
    rc = imgToSee_4p(:,:,1);
    rc(border_4p ==1) = cm(1,1);
    gc = imgToSee_4p(:,:,2);
    gc(border_4p ==1) = cm(1,2);
    bc = imgToSee_4p(:,:,3);
    bc(border_4p ==1) = cm(1,3);
    imgToSee_4p(:,:,1) = rc;
    imgToSee_4p(:,:,2) = gc;
    imgToSee_4p(:,:,3) = bc;
    
    % ref mask
    if k<=cn_ref
        comp_ref = A_ref(:,:,k)' ;
        mask_ref = imbinarize(comp_ref, 0.01);
        border_ref = imdilate(mask_ref, bbb) - imerode(mask_ref, bbb);
        
        rc = imgToSee_ref(:,:,1);
        rc(border_ref ==1) = cm(2,1);
        gc = imgToSee_ref(:,:,2);
        gc(border_ref ==1) = cm(2,2);
        bc = imgToSee_ref(:,:,3);
        bc(border_ref ==1) = cm(2,3);
        imgToSee_ref(:,:,1) = rc;
        imgToSee_ref(:,:,2) = gc;
        imgToSee_ref(:,:,3) = bc;
        
    end
    
    % basic mask
    rc = imgToSee_gt(:,:,1);
    rc(border_4p ==1) = cm(1,1);
    gc = imgToSee_gt(:,:,2);
    gc(border_4p ==1) = cm(1,2);
    bc = imgToSee_gt(:,:,3);
    bc(border_4p ==1) = cm(1,3);
    imgToSee_gt(:,:,1) = rc;
    imgToSee_gt(:,:,2) = gc;
    imgToSee_gt(:,:,3) = bc;
    
%     rc = imgToSee_gt(:,:,1);
%     rc(border_ref ==1) = cm(2,1);
%     gc = imgToSee_gt(:,:,2);
%     gc(border_ref ==1) = cm(2,2);
%     bc = imgToSee_gt(:,:,3);
%     bc(border_ref ==1) = cm(2,3);
%     imgToSee_gt(:,:,1) = rc;
%     imgToSee_gt(:,:,2) = gc;
%     imgToSee_gt(:,:,3) = bc;
    
    figure(1)
    subplot(1,3,1);imshow(imgToSee_gt,[]);
    subplot(1,3,2);imshow(imgToSee_4p,[]);
    subplot(1,3,3);imshow(imgToSee_ref,[]);
    title(num2str(k))
    
    if k == cn_4p
        break
    end
    
end
