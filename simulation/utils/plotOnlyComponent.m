function rlt = plotOnlyComponent(A)
% plot spatial components on  the SD image

[xs,ys,~]=size(A);
baseimg = uint8(ones(ys,xs)) *255;  % white background
% baseimg = uint8(ones(xs,ys)) ;      % black background

color=[255 47 39];

[xs,ys]=size(baseimg);

rc = baseimg;
gc = baseimg;
bc = baseimg;
for i = 1:size(A, 3)
    comp = A(:,:,i)';
    border = getBorder(comp, 0.2);
    
    
    % change color
    rc(border)=color(1,1);
    gc(border)=color(1,2);
    bc(border)=color(1,3);
    
end

baseimg_c = uint8(zeros(xs,ys,3));
baseimg_c(:,:,1) = rc; baseimg_c(:,:,2) = gc; baseimg_c(:,:,3) = bc;
imshow(baseimg_c)
rlt = baseimg_c;
end