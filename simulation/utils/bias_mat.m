function bias = bias_mat(C, scale)

[N, T] = size(C);
bias = scale * (0:N-1)';
bias = repmat(bias, 1 , T);

end