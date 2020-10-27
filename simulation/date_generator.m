%% Autocalibration demo
% This script demonstrates on simulated data how to use the MLspike and
% autocalibration algorithms. To see how to run the algorithms, go directly
% to section 'Spike estimation'.

%% Generate spikes
% We first generate some simulated data. Spike trains are generated using
% spk_gentrain function (the data consists of 6 trials of 30s).
%
% Note the usefull function spk_display to display both spikes and calcium
% time courses.
clc;clear;close all;

%% add dependencies
addpath('./spikes/')
addpath('./brick/')
addpath('./brick/private/')
addpath('./utils/')

%% parameters
n_neuron = 120;
T = 150;
rate = 0.4;

%% generate spikes
spikes = spk_gentrain(rate,T,'bursty','repeat',n_neuron);

% display
% figure(1)
% spk_display([],spikes,[])

%% Random A and tau, and random level of noise
% We draw randomly the parameters governing the calcium dynamics A (DF/F
% amplitude for one spike) and tau (decay time constant), as well as the
% level of noise in the data, sigma.

% DF/F
amin = 0.1;
amax = 0.25;
a = amin * exp(rand(1)*log(amax/amin)); % log-uniform distribution

% decay time
taumin = 0.3;
taumax = 1.0;
tau = taumin * exp(rand(1)*log(taumax/taumin));

% no noise
sigmamin = 1e-6;
sigmamax = 1e-6;
sigma = sigmamin * exp(rand(1)*log(sigmamax/sigmamin));

%% Generate calcium, GCaMP6f dynamics
% We generate calcium signals corresponding to the above spike trains using
% these parameters. Note that some additional parameters (OGB dye
% saturation and drift parameter) are fixed.

% parameters
dt = .033; % 30Hz (video-rate) acquisition
pcal = spk_calcium('par');
pcal.dt = dt;
pcal.a = a;
pcal.tau = tau;
pcal.saturation = .1; % saturation level is fixed
pcal.pnonlin = [0.5000 0.0100];
pcal.sigma = sigma; % noise level
pcal.drift.parameter = [0.0 0.0]; % drift level (#harmonics and amplitude of them)

% generate calcium
calcium = spk_calcium(spikes,pcal);

% display
% figure(1)
% spk_display(dt,spikes,calcium)
% drawnow

%% generate (ground truth) video
% We generate calcium imaging videos based on randomly selected neuron
% spatial profiles. 520 um FOV.

N = floor(1/dt)*T;
% rescale and plot calcium profile
calcium_all = [];
for i = 1:length(calcium)
    c = calcium{i};
    c = c-min(c(:));
    c = c/max(c(:));
    calcium{i} = c + 0.1;
    calcium_all = [calcium_all; c(1:N)'];
end
calcium_all = line_norm(calcium_all);
calcium_all = calcium_all + bias_mat(calcium_all, 1);
figure; plot(calcium_all')
pause(1)

% load and normlize template
template = load('./utils/resV1_trs.mat');
A = full(template.A);
A = reshape(A, [256,256,size(A, 2)]);
A = imresize3(A, [512,512,size(A, 3)]);
A = A/max(A(:)); % normalize A

% select spatial profiles randomly
rand_idx = randperm(size(A, 3), n_neuron);

% make video
calcium_video = zeros(size(A,1), size(A,1), N);
for j = 1:N % make video
    fprintf('Generating video: %d of %d...\n', j, N)
    
    for i = 1: n_neuron % make frame
        
        idx = rand_idx(i);
        spatial_ = A(:,:,idx);
        spatial_ = spatial_ / max(spatial_(:));
        
        calcium_video(:,:,j) = calcium_video(:,:,j) + spatial_ .* calcium{i}(j);
    end
    
end


%% save
fprintf('Saving...\n')
calcium_video = calcium_video/max(calcium_video(:));
calcium_video = uint16(calcium_video * 65535*0.8) + 0.2*65535;

saveastiff(calcium_video, 'calcium_video_30Hz_dxy_1um_test.tif')


%% Add noise
calcium_video_mpg = mpg_noise(calcium_video, 7, 1000);
saveastiff(calcium_video_mpg, 'calcium_video_30Hz_dxy_1um_MPG.tif')
