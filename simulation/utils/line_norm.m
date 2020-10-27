function C_norm = line_norm(C)
% line-wise normalization
%
[~, T] = size(C);
c_min = min(C')';
c_min_mat = repmat(c_min, 1,T);
C = C-c_min_mat;

c_max = max(C')';
c_max_mat = repmat(c_max, 1,T);
C_norm = C ./ c_max_mat;

end