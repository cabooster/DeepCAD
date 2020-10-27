function [bordermask, innermask] = getBorder(comp, theroshold)
% 
% return border of an image of an single connected area
%
    comp = uint8(comp/max(comp(:)) * 255);
    imbi = imbinarize(comp);
    border = bwboundaries(imbi);border=border{1};
    bordermask = logical(zeros(size(comp)));
    for i = 1:size(border,1)
        bordermask(border(i,1), border(i,2)) = 1;
    end
    bordermask = imdilate(bordermask, ones(2,2));
    
    innermask = imbi;
end