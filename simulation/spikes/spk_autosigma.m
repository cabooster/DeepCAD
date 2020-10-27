function varargout = spk_autosigma(varargin)
% function psig = spk_autosigma('par'[,presetflag])
% function sigmaest = spk_autosigma(calcium,dt[,psig|presetflag])
%---
% Estimation of the level of noise in the calcium signal.
%
% Input:
% - calcium     numerical vector or cell array thereof - calcium signal(s)
% - dt          frame acquisition time
% - presetflag  'white', 'correlated' or 'correlatedbias'
%
% Input/Output:
% - psig        structure with parameters 'freqs', 'bias' and 'donormalize'
%               (see below) 
% 
% Output:
% - sigmaest    estimated noise level in the normalized calcium signal
%               (i.e. in calcium/mean(calcium)
%
% The noise level is estimated in such a way that, if calcium consists of a
% pure white noise of RMS sigma, then sigmaest = bias * sigma. 
% However, the input 'calcium' is generally not a pure white noise (it
% consists of some true signal plus some noise, that itself might be
% temporally correlated). Therefore, parameter 'freqs' denotes a frequency
% band where the noise RMS can be estimated, then the estimation is
% obtained by extrapolation to the full range of frequencies present in the
% data, according to the formula: 
% sigmaest = bias * RMS(calcium filtered in 'freqs') * sqrt(extent of 'freqs'/extent of all frequencies)
%
% Parameter 'freqs' can be either:
% - a 2-elements vector [fmin fmax] (a band-pass filter will be used)
% - a scalar fmin (a high-pass filter will be used)
% - the string 'diff' (the derivative - which is a convenient high-pas filter - will be used)
%
% If parameter 'donormalize' is set to true (this is the default), calcium
% is replaced by calcium/mean(calcium) before noise level estimation.

% Default parameter
if ischar(varargin{1})
    if ~strcmp(varargin{1},'par'), error argument, end
    par = defaultpar(varargin{2:end});
    varargout = {par};
    return
end

% Auto-calibration
calcium = varargin{1};
dt = varargin{2};
if nargin>=3
    psig = defaultpar(varargin{3:end});
else
    psig = defaultpar;
end
varargout = {autosigma(calcium,dt,psig)};
    
%---
function psig = defaultpar(varargin)

psig = struct;
psig.freqs = [3 20]; % = preset 'correlated'
psig.bias = 1;
psig.donormalize = false;

if nargin==0, return, end

if ischar(varargin{1}) && fn_ismemberstr(varargin{1},{'white' 'correlated' 'correlatedbias'})
    presetflag = varargin{1};
    varargin(1) = [];
    switch presetflag
        case 'white'
            psig.freqs = 3;
        case 'correlated'
            psig.freqs = [3 20];
        case 'correlatedbias'
            psig.freqs = [3 20];
            psig.bias = 0.6894;
    end
end

if isempty(varargin)
    % nothing to do
elseif isstruct(varargin{1})
    psig = fn_structmerge(psig,varargin{1});
else
    psig = fn_structmerge(psig,struct(varargin{:}),'strict');
end

%---
function sigmaest = autosigma(calcium,dt,psig)

if ~iscell(calcium), calcium = {calcium}; end
ntrial = length(calcium);
if isscalar(dt), dt = repmat(dt,[1 ntrial]); end
sigmaest = zeros(1,ntrial);
for k=1:ntrial
    x = calcium{k};
    if psig.donormalize, x = x/mean(x); end
    dtk = dt(k); ntk = length(x);
    if strcmp(psig.freqs,'diff')
        sigmaest(k) = rms(diff(x))/sqrt(2);
    else
        fnyquist = 1/(2*dtk);
        if isscalar(psig.freqs), psig.freqs(2) = fnyquist; end
        xf = fft(x)/sqrt(ntk);
        freqs = abs(fn_fftfrequencies(ntk,1/dtk,'centered'));
        okfreq = (freqs>=psig.freqs(1) & freqs<=psig.freqs(2));
        sigmaest(k) = rms(xf(okfreq));
    end
end
sigmaest = psig.bias * mean(sigmaest);
