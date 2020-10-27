function img_MPG = mpg_noise(img, quantum_well_depth, sigma_read)

% ----- Pixel Well Depth ----- %
% quantum_well_depth: in photoelectrons

% ----- Image ----- %
img = double(img);
img_photoelectrons = img*quantum_well_depth/65535;
% ----------------- %

% ----- Shot Noise ----- %
img_poisson = poissrnd(img_photoelectrons);

% ---------------------- %
img_poisson = img_poisson/quantum_well_depth*65535;

% ----- Read Noise ----- %
% sigma_read: photoelectrons RMS
img_MPG = uint16(img_poisson + sigma_read*randn(size(img_poisson)));

end
