function rlt = change_background_white(A)
% a function to change background into white
% A: a RGB colored image, uint8
% rlt: result image
%
[xs ,ys, zs] = size(A);
mask=max(A,[],3);
thr = 100;
rlt = uint8(ones(xs ,ys, zs));
r = A(:,:,1);
r(mask<thr) =255;
g = A(:,:,2);
g(mask<thr) =255;
b = A(:,:,3);
b(mask<thr) =255;

rlt(:,:,1)=r;
rlt(:,:,2)=g;
rlt(:,:,3)=b;


end