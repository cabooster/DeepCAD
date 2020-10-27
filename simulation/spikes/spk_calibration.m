function [pest fit drift rcalcium] = spk_calibration(spikes,F,varargin)
% function [pest fit drift rcalcium] = spk_calibration(spikes,F,pcal|dt[,other parameters])
% function pcal = spk_calibration('par'[,dt][,'drift|trend'][,other parameters])
% function [pest fit drift] = spk_calibration(spikes,F,pcal,pest) % no estimation, output fit and drift only
%---
% estimate a, tau, saturation
%
% Input:
% - spikes      real spikes
% - F           corresponding fluorescence signals
%               if pcal.doglobal is set to true, F should be a 2*n cell
%               array, with both signals inside the neuron and inside the
%               surrounding neuropil
% - pcal        parameters for the calibration; fields are:
%               .dt      sampling time
%               .tdrift  time constant of the low-frequency drifts (or
%                        'trend' if there is only a linear trend)
%               .dosaturation, .dodelay, .doglobal, .dohill, .doc0, .doton
%                       any saturation / delay / global signal etc. in the model
%
% Output:
% - pest        estimated parameters
% - fit         calcium fit based on real spikes
% - drift       drift part of this fit
% - rcalcium    "real" calcium (where accurate amount of global signal has
%               been subtracted)

if nargin==0, help spk_calibration, return, end
 
% Default parameter
if ischar(spikes)
    if ~strcmp(spikes,'par'), error argument, end
    if nargin>=2, varargin = [F varargin]; end
    pest = defaultpar(varargin{:});
    return
end

% Input
% (handle multiple data)
if ~iscell(spikes)
    if isvector(spikes), spikes = {spikes(:)}; else spikes = shiftdim(num2cell(spikes,1)); end
end
Fiscell = iscell(F);
if ~Fiscell
    if isvector(F), F = {F(:)}; else F = shiftdim(num2cell(F,1)); end
end
ndata = numel(spikes);
for itest=1:ndata, F{itest} = double(F{itest}); end
% calibration parameters
if nargin>=3 && isstruct(varargin{1})
    pcal = fn_structmerge(defaultpar,varargin{1},'skip');
else
    pcal = defaultpar(varargin{:});
end
% split F into cell and global calcium signals
if isvector(F) && ~pcal.doglobal, F = row(F); end
if any(size(F)~=[(1+pcal.doglobal) ndata])
    error 'spike data and fluorescence data do not match'
end
if pcal.doglobal
    cglobal = F(2,:);
    F = F(1,:);
else
    cglobal = [];
end
% no estimation, output fit and drift only?
noest = (nargin>=4 && isstruct(varargin{2}));
if noest, pest = varargin{2}; end

% forward parameters
pfwd = spk_calcium('par');
pfwd.dt = pcal.dt;
nt = fn_map(@length,F);
pfwd.T = nt.*pcal.dt;

% estimation
if ~noest
    variables = {'a' 'tau' 'saturation' 'delay' 'contamination' 'p2' 'p3' 'hill' 'c0' 'ton'};
    doest = [true true pcal.dosaturation pcal.dodelay pcal.doglobal pcal.dononlinear pcal.dononlinear pcal.dohill pcal.doc0 pcal.doton];
    if ~strcmp(pcal.display,'none')
        disp(fn_strcat(variables(doest),'CALIBRATION: estimate ',', ',''))
    end
    
    opt = optimset('Algorithm','interior-point', ... note that previous choice of 'active-set' was sometimes getting stuck in "not even local minima"!!
        'maxfunevals',10000,'tolx',1e-20,'tolfun',1e-5, ...
        'display',fn_switch(pcal.display,'debug','iter',pcal.display)); %,'PlotFcns',{@optimplotx,@optimplotfval,@optimplotstepsize,@optimplotconstrviolation});
    % parameters: a, tau, saturation, delay, global contribution, p2, p3, hill, c0, ton
    pstart = [.1 1 -2 0 .5 0 0 1 .5 .005]; % we estimate the log of the saturation
    LB = [0 .1 -4 -.05 0 -5 -5 .5 0 0 0];
    UB = [1.3 3 log10(0.2) .05 1 5 5 4 2 .1];
    FACT = [1e-2 1e-2 1e-2 1e-4 1e-2 1e-1 1e-1 1e-2 1e-2 1e-2];
    pstart=pstart(doest); LB=LB(doest); UB=UB(doest); FACT=FACT(doest);
    
    pvalues = fmincon(@(p)energycalib1(p./FACT,spikes,F,cglobal,pfwd,pcal),pstart.*FACT,[],[],[],[],LB.*FACT,UB.*FACT,[],opt)./FACT;
else
    pvalues = [pest.a pest.tau log10(pest.saturation)];
end

% gather the results
pest = tps_mlspikes('par');
pest.dt = pcal.dt;
[e fit drift rcalcium] = energycalib1(pvalues,spikes,F,cglobal,pfwd,pcal); %#ok<ASGLU>
fit = reshape(fit,size(F));
if ~Fiscell
    fit = [fit{:}];
    drift = [drift{:}];
    rcalcium = [rcalcium{:}];
end
pest.F0 = [];
pest.a = pvalues(1);
pest.tau = pvalues(2);
kp = 2;
if pcal.dosaturation, kp=kp+1; pest.saturation = 10^pvalues(kp); end
if pcal.dodelay, kp=kp+1; pest.delay = pvalues(kp); end
if pcal.doglobal, kp=kp+1; pest.contamination = pvalues(kp); end
if pcal.dononlinear, pest.pnonlin = pvalues(kp+(1:2)); kp = kp+2; end
if pcal.dohill, kp=kp+1; pest.hill = pvalues(kp); end
if pcal.doc0, kp=kp+1; pest.c0 = pvalues(kp); end
if pcal.doton, kp=kp+1; pest.ton = pvalues(kp); end

%---
function [e fit drift F] = energycalib1(p,spikes,F,cglobal,pfwd0,pcal)

% parameters and sizes
ndata = numel(spikes);
nt = fn_map(@length,F,'array');
pfwd0.a = p(1);
pfwd0.tau = p(2);
kp=2;
if pcal.dosaturation
    kp=kp+1;
    pfwd0.saturation = 10^p(kp); 
else
    pfwd0.saturation = 0; 
end
if pcal.dodelay
    kp=kp+1;
    spikes = fn_map(@(u)u+p(kp),spikes);
end
if pcal.doglobal
    kp=kp+1;
    r = p(kp);
    for i=1:ndata, F{i} = F{i} - mean(F{i})*((cglobal{i}-1)*r); end
end
if pcal.dononlinear, pfwd0.pnonlin = p(kp+(1:2)); kp=kp+2; end
if pcal.dohill, kp=kp+1; pfwd0.hill = p(kp); end
if pcal.doc0, kp=kp+1; pfwd0.c0 = p(kp); end
if pcal.doton, kp=kp+1; pfwd0.ton = p(kp); end

if isscalar(pfwd0.dt), pfwd0.dt=repmat(pfwd0.dt,[1 ndata]); end
if isscalar(pfwd0.T), pfwd0.T=repmat(pfwd0.T,[1 ndata]); end

% forward prediction
Fpred0 = cell(1,ndata);
for i=1:ndata
    pfwd = pfwd0;
    pfwd.dt = pfwd0.dt(i);
    pfwd.T = pfwd0.T(i);
    Fpred0{i} = spk_calcium(spikes{i},pfwd);
end

% estimate baseline and prediction including drift
drift = cell(1,ndata);
fit = cell(1,ndata);
dif = cell(1,ndata);
for i=1:ndata
    base = F{i}./Fpred0{i};
    if isequal(pcal.tdrift,0)
        drift{i} = mean(base)*ones(nt,1);
    elseif strcmp(pcal.tdrift,'trend')
        drift{i} = base - detrend(base);
    else
        drift{i} = fn_filt(base,pcal.tdrift/pfwd0.dt(i),'lmd',1); 
    end
    fit{i} = drift{i}.*Fpred0{i};
    dif{i} = F{i}-fit{i};
    if pcal.tsmooth % smooth the difference 
        dif{i} = fn_filt(dif{i},pcal.tsmooth/pfwd0.dt(i),'lmd',1);
    end
end

% error
e = sqrt(mean(cat(1,dif{:}).^2))*100;

if strcmp(pcal.display,'debug')
    fprintf('%.6f ',p)
    fprintf('-> %.6f\n',e)
end

%---
function [e nest Fest] = energycalib2(logspikerate,spikereal,F,par,cost)

par.spikerate = 10^logspikerate;
fprintf('\ntrying spike rate = %.1g\n',par.spikerate)
s = size(F);
ndata = numel(F);
nest = cell(s); Fest = cell(s); eval = zeros(s);
for i=1:ndata
    [nest{i} Fest{i}] = tps_mlspikes(F{i},par);
    spikeest = fn_timevector(nest{i},par.dt,'times');
    eval(i) = spk_distance(spikereal{i},spikeest,cost);
end
e = sum(eval);

%-------------------------------------------------------------------------%
%                       CALIBRATION PARAMETERS                            %
%-------------------------------------------------------------------------%

function pcal = defaultpar(varargin)

% Mandatory parameter
pcal.dt = [];

% Parameters to estimate
pcal.dosaturation = true;
pcal.dononlinear = false;
pcal.dohill = false;
pcal.doc0 = false;
pcal.dodelay = false;
pcal.doton = false;
pcal.doglobal = false;

% Drift
pcal.tdrift = 0; % time constant for drift

% Energy
pcal.tsmooth = 0; % smoothing before computing the energy to be minimized

% Display
pcal.display = 'iter'; % 'none', 'final', 'iter' or 'debug'

% User input
narg = length(varargin);
if narg==0, return, end
p = struct;
i = 0;
while i<length(varargin)
    i = i+1;
    a = varargin{i};
    if isnumeric(a)
        pcal.dt = a;
    else
        switch a
            case 'drift'
                pcal.tdrift = 5; % default drift time constant of 2s
            case 'trend'
                pcal.tdrift = 'trend'; % linear drift
            otherwise
                i = i+1;
                b = varargin{i};
                sub = regexp(a,'(.*)\.(.*)','tokens');
                if isempty(sub)
                    p.(a) = b;
                else
                    p.(sub{1}{1}).(sub{1}{2}) = b;
                end
        end
    end
end
pcal = fn_structmerge(pcal,p,'strict','recursive');


%-------------------------------------------------------------------------%
%                       TOOLS                                             %
%-------------------------------------------------------------------------%

function A = builddrifts(nt,dt,tdrift)

T = dt*nt;
ndrift = 1+round(T/tdrift);

A = linspace(-1,1,nt)';
if ndrift>1
    phase = linspace(0,2*pi,nt)';
    nsin = (ndrift-1)/2;
    A(1,1+2*nsin) = 0; % pre-allocate
    for k=1:nsin
        A(:,2*k) = sin(k*phase);
        A(:,2*k+1) = cos(k*phase);
    end
end

if ~matrixonly
    A = A * a(:);
end
