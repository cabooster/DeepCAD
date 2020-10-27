function varargout = tps_mlspikes(varargin)
% function [n Ffit par LL xest|PP drift] = tps_mlspikes(F,par|dt[,other parameters]) = estimate spikes from calcium fluorescence
% function Fpred = tps_mlspike(n,par);                        = predict calcium from spikes
% function par = tps_mlspike('par'[,dt][,other parameters])   = get default parameters
%---
% 
% Input/Output:
% - F       column vector, array (one signal per column) or cell array of
%           column vectors - fluorescence signal 
% - n       same size as F - vectors of spike counts
% - dt      scalar - frame acquisition time
% - par     parameter structures - fields are:
%           .dt     frame acquisition time
%           .F0     baseline fluorescence (use [] to estimate it)
%           .a      amplitude of 1 spike
%           .tau    decay time
%           .spikerate
%           .drift  subfield 'method' is 'state' or 'basis functions'
%           it is possible to provide different parameters for each
%           individual calcium time courses by using a non-scalar structure
%
% Output:
% - P       posterior probability of intracellular calcium at each instants
% - LL      negative log-likelihood of the measure

if nargin==0, help tps_mlspikes, return, end
 
% Get default parameters
if ischar(varargin{1})
    if ~strcmp(varargin{1},'par'), error argument, end
    [par spec] = defaultpar(varargin{2:end});
    varargout = {par};
    if nargout>=2, varargout{2} = spec; end
    return
end

% Input calcium [backward] or spikes [forward]; handle multiple data
x = varargin{1};
xiscell = iscell(x);
if ~xiscell && all(size(x)>1)
    x = num2cell(x,1); 
end
if iscell(x)
    ndata = numel(x);
else
    ndata = 1;
end

% Parameters; handle several sets of parameters as well
if isstruct(varargin{2})
    par = varargin{2};
    if isscalar(par)
        dt = par.dt;
        if isempty(dt), error 'sampling time not defined', end
        if ndata>1
            if isscalar(dt), dt = repmat(dt,1,ndata); end
            par = repmat(par,1,ndata);
            for i=1:length(par), par(i).dt = dt(i); end
        end
    end
    par = num2cell(par);
    for i=1:ndata
        par{i} = fn_structmerge(defaultpar,par{i},'strict','recursive');       
    end
    par = cell2mat(par);
else
    dt = varargin{2};
    if length(dt)~=ndata, error 'length mismatch', end
    for i=1:ndata
        par(i) = defaultpar(dt(i),varargin{3:end}); %#ok<AGROW>
    end
end
displaymode = par(1).display;
if strcmp(displaymode,'default')
    displaymode = fn_switch(ndata==1,'steps','count');
    [par.display] = deal(displaymode);
end

% Autocalibration of parameter finetune.sigma? (= a priori level of noise)
finetune = [par.finetune];
sigmas = {finetune.sigma};
doautosigma = fn_isemptyc(sigmas);
if any(diff(doautosigma)), error 'multiple data: parameter finetune.sigma must be always empty or always non-empty', end
if doautosigma(1)
    if ndata>1 && ~isequal(par.dt), error 'cannot auto-estimate finetune.sigma on multiple data with different time constants', end
    if ndata>1 && ~isequal(finetune.autosigmasettings), error 'cannot auto-estimate finetune.sigma on multiple data with different estimation settings', end
    sigmaest = spk_autosigma(x,par(1).dt,finetune(1).autosigmasettings);
    for i=1:ndata, par(i).finetune.sigma = sigmaest; end
end

% Multiple data
if iscell(x)
    nout = max(1,nargout);
    out = cell(nout,ndata);
    switch displaymode
        case 'steps'
            fn_progress('tps_mlspike',ndata,'noerase')
        case 'count'
            fn_progress('tps_mlspike',ndata)
        case 'none'
    end
    for k=1:ndata
        if ~strcmp(displaymode,'none'), fn_progress(k), end
        [out{:,k}] = tps_mlspikes(x{k},par(k)); 
    end
    varargout = num2cell(out,2);
    if xiscell, icat = 3:min(4,nout); else icat = 1:nout; end
    for i=icat, varargout{i} =  [varargout{i}{:}]; end
    return
end

% Backward (estimate spikes) or forward (generate fluorescence)?
if any(isnan(x))
    % we assume here that user is trying to compute spikes from a
    % non-defined time courses
    out = {[] nan(size(x)) []};
    varargout = out(1:max(1,nargout));
elseif sum(x==0)>length(x)/2
    % Predict calcium fluorescence (input is a spike train)
    n = x;
    Fpred = forward(n,par);
    varargout = {Fpred};
else
    % Estimate spikes (input is a fluorescence signal)
    F = x;
    varargout = cell(1,max(1,nargout));
    [varargout{:}] = backward(F,par);
end

%-------------------------------------------------------------------------%
%                       PARAMETERS                                        %
%-------------------------------------------------------------------------%

function [par spec] = defaultpar(varargin)

% Mandatory parameter
spec.acquisition__time__must__be__set = 'label';
par.dt = [];
spec.dt = 'double';

% Physiological parameters
spec.set__baseline__value__only__if__known = 'label';
par.F0 = [];
par.a = .1;
par.tau = 1;
par.ton = 0;
par.saturation = 0;
par.pnonlin = [];
par.hill = 1;
par.c0 = 0; % calcium baseline level in "individual spike transient amplitude" unit; needs to be set only when par.hill~=1
spec.F0 = 'xdouble';
spec.physiological__parameters = 'label';
spec.a = 'double';
spec.tau = 'double';
spec.ton = 'double';
spec.saturation = 'double';
spec.pnonlin = 'double';
spec.hill = 'double';
spec.c0 = 'double';

% Drifts
spec.drift__parameters = 'label';
par.drift.effect = 'multiplicative';
par.drift.parameter = 0;
par.drift.baselinestart = false;
spec.drift = struct( ...
    'preffered__method__is__state', 'label', ...
    'effect',   {{'multiplicative' 'additive'}}, ...
    'set__positive__parameter__to__estimate__drifts', 'label', ...
    'parameter',        'double', ...
    'impose__resting__activity__at__startup', 'label', ...
    'baselinestart',    'logical' ...
    );

% Fine tuning 
spec.fine__tuning__parameters = 'label';
par.finetune.spikerate = .1;
par.finetune.sigma = [];
par.finetune.autosigmasettings = 'correlated';
% par.finetune.sigmaestsmooth = 2; % sigma is estimated as the std of the high-passed signal (unit: second)
% par.finetune.sigmaestboost = 1; % it was found heuristically that it is good that sigma be slightly overestimated
spec.finetune = struct( ...
    'a__priori__level__of__spiking', 'label', ...
    'spikerate',    'double', ...
    'set__STD__of__noise__only__if__known', 'label', ...
    'sigma',        'xdouble', ...
    'otherwise__use__predefined__setting__for__auto__estimation', 'label', ...    
    'autosigmasettings',   {{'white' 'correlated' 'correlatedbias'}} ...
    );

% Algorithm parameters
spec.algorithm__private__parameters = 'label';
par.algo.estimate = 'MAP'; % 'MAP', 'proba' or 'samples'
par.algo.cmax = 10;
par.algo.nc = []; % use appropriate default (see below) if not set
par.algo.nc_norise = 100;
par.algo.nc_rise = 50;
par.algo.nb = []; % use appropriate default (see below) if not set
par.algo.nb_nodrift = 40;
par.algo.nb_driftstate = 100;
par.algo.nb_driftrise = 50;
par.algo.smax = .1; % 10% DF/F in one second, for 'driftslope' only
par.algo.ns = 21; % for 'driftslope' only
par.algo.np = 50; % for 'driftrise' only
par.algo.nsample = [];
par.algo.interpmode = 'spline'; % 'linear' or 'spline'; choosing one or 
% the other slightly changes the result, 'spline' seems better in general,
% but not always (in particular when grid is too coarse 'linear' might be
% better?)
par.algo.testflag = 0; % use this for some debugging
spec.algo = struct( ...
    'return__a__unique__spike__train__or__probabilities__or__samples', 'label', ...
    'estimate',     {'MAP' 'proba' 'samples'}, ...
    'increase__cmax__if__estimation__seems__to__saturate', 'label', ...
    'cmax', 'double', ...
    'sampling__factors__affect__speed__and__performance', 'label', ...
    'nc',   'double', ...
    'nb',   'double', ...
    'nb_nodrift',       'double', ...
    'nb_driftstate',    'double', ...
    'nb_driftrise',    'double', ...
    'for__slope__drift__method__only', 'label', ...
    'smax', 'double', ...
    'ns',   'double', ...
    'np',   'double', ...
    'nsample',      'double', ...
    'interpmode',   {{'linear' 'spline'}}, ...
    'unused__parameters', 'label', ...
    'testflag',     'logical' ...
    );

% Special
spec.special__behavior__parameters = 'label';
par.special.nonintegerspike_minamp = 0; % allow non-integer spike and set the minimal amplitue (used for auto-calibration of a)
par.special.burstcostsone = false;
spec.special = struct( ...
    'detect__calcium__events__with__continuous__amplitude__range',  'label', ...
    'nonintegerspike_minamp',   'double', ...
    'favor__multiple__simultaneous__spikes', 'label', ...
    'burstcostsone',     'logical' ...
    );

% Display
spec.display__parameters = 'label';
par.display = 'default'; % possibilities are 'none', 'count', 'steps' and 'default' (=auto-decide between 'count' and 'steps')
par.dographsummary = true;
spec.display = {'default' 'none' 'count' 'steps'};
spec.dographsummary = 'logical';

% User input
narg = length(varargin);
if narg==0, return, end
p = struct;
i = 0;
while i<length(varargin)
    i = i+1;
    a = varargin{i};
    if isnumeric(a)
        par.dt = a;
    else
        switch a
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
par = fn_structmerge(par,p,'strict','recursive');

%-------------------------------------------------------------------------%
%                       FORWARD                                           %
%-------------------------------------------------------------------------%

function Fpred = forward(n,par)

dt = par.dt;
a = par.a;
tau = par.tau;
ton = par.ton;
sat = par.saturation;
pnonlin = par.pnonlin;
hill = par.hill;
c0 = par.c0;
decay = exp(-dt/tau);
T = length(n);

if length(n)~=T % time of spikes rather than number of spikes!
    spikes = n;
    n = zeros(1,T);
    if isfield(par,'t0'), t0=par.t0; else t0=0; end
    for k=1:length(spikes)
        idx = max(1,min(T,1+round((spikes(k)-t0)/dt)));
        spikes(idx) = spikes(idx)+1;
    end
end

% calcium: convolution with unit exponential
c = zeros(T,1);
ct = 0;
for t=1:T
    ct = ct*decay + n(t);
    c(t) = ct;
end

% probe
if hill==1
    cn = c;
else
    cn = (c0+c).^hill-c0^hill;
end
p = cn ./ (1+sat*cn);
if ton>0
    ptarget = p;
    pspeed = (1+sat*cn)/ton;
    p = zeros(T,1); 
    pt = 0;
    for t=1:T
        pt = ptarget(t) + (pt-ptarget(t))*exp(-pspeed(t)*dt);
        p(t) = pt;
    end
end

% ad-hoc nonlinearity
if ~isempty(pnonlin)
    p = polyval([fliplr(pnonlin) 1-sum(pnonlin) 0],p);
end

% measure
ypred = 1 + a*p;

% add drifts
if isfield(par.drift,'method') && strcmp(par.drift.method,'basis functions') && isfield(par.drift,'estimate')
    switch par.drift.effect
        case 'additive'
            ypred = ypred + builddrifts(par.drift.estimate,T);
        case 'multiplicative'
            ypred = ypred .* (1+builddrifts(par.drift.estimate,T));
        otherwise
            error flag
    end
end

% scale by baseline
if ~isempty(par.F0)
    F0 = par.F0;
else
    F0 = 1;
end
Fpred = ypred*F0;

%-------------------------------------------------------------------------%
%                       BACKWARD                                          %
%-------------------------------------------------------------------------%

% general backward function
function [n Ffit par LL xest drift]= backward(F,par)

% input
F = double(F(:));

% some parameter checks
if ~any(strcmpi(par.algo.estimate,{'MAP' 'proba' 'sample' 'samples'}))
    error 'parameter algo.estimate should be either ''MAP'' (return a unique spike train), ''proba'' (return spike probabilities) or ''samples'' (return algo.nsample sample spike trains)'
end
if any(strcmpi(par.algo.estimate,{'sample' 'samples'})) 
    if isempty(par.algo.nsample)
        disp 'number of samples not defined in algo.nsample, using default value 200'
        par.algo.nsample = 200;
    elseif par.algo.nsample<2
        error 'algo.nsample must be at least 2'
    end
end

% minimum and maximum values for baseline
if isempty(par.F0)
    if par.drift.parameter
        par.F0 = [min(min(F),mean(F)/1.5) prctile(F,90)];
    else
        par.F0 = [min(min(F),mean(F)/1.5) max(mean(F),median(F))];
    end
    if par.F0(1)<=0
        if strcmp(par.display,'steps'), fprintf('min F0 %.3f',par.F0(1)), end
        par.F0(1)=par.F0(2)/100;
        if strcmp(par.display,'steps'), fprintf('-> %.3f\n',par.F0(1)), end
    end
end

% Run the appropriate estimation function
doMAP = strcmpi(par.algo.estimate,'MAP');
dodrift = (par.drift.parameter>0);
dobaseline = ~isscalar(par.F0);
if dodrift && ~dobaseline, error 'parameters specify a drift estimation, but at the same time a fixed baseline!', end
dorise = (par.ton>0);
if dorise
    if (~dobaseline || ~dodrift)
        error 'rise time only implemented together with a drifting state'
    end
    if doMAP
        [n Ffit par.F0 LL xest] = maxposterior_driftstaterise(F,par);
    else
        error 'rise time only implemented for MAP estimations'
    end
elseif ~dobaseline
    par.algo.nb = 1;
    par.F0 = par.F0*[1 1];
    [n Ffit par.F0 LL xest] = maxposterior_fixbaseline(F,par);
elseif ~dodrift
    % no drift; first estimate F0 by trying a number of different baselines
    if strcmp(par.display,'steps'), disp 'estimate fix baseline', drawnow, end %#ok<DUALC>
    [n Ffit par.F0 LL xest] = maxposterior_fixbaseline(F,par);
else
    if strcmp(par.display,'steps'), disp 'estimate drifting baseline', end
    [n Ffit par.F0 LL xest] = maxposterior_driftstate(F,par);
end    

% "Assemble" the drift
if dodrift
    if dorise
        drift = squeeze(xest(:,3,:))*par.F0;
    else
        drift = squeeze(xest(:,2,:))*par.F0;
    end
else
    nt = length(F);
    drift = ones(nt,1)*par.F0;
end

% display
if par.dographsummary
    ha = initgraphsummary;
    n1 = fn_switch(par.special.nonintegerspike_minamp>0,round(n*par.a*100),n); 
    if doMAP
        spk_display(par.dt,n1,{F Ffit drift},'in',ha(4))
    elseif strcmpi(par.algo.estimate,'proba')
        spk_display(par.dt,{[] n1},{F Ffit drift},'in',ha(4),'rate')
    else % samples
        spk_display(par.dt,n1(:,1),{F Ffit(:,1) drift(:,1)},'in',ha(4))
        ylabel(ha(4),['sample 1/' num2str(size(n1,2))])
    end
    drawnow
end

%-------------------------------------------------------------------------%
% no drift, test one or multiple baseline values
function [n Ffit F0 LL xest yfit]= maxposterior_fixbaseline(F,par)

DEBUG = eval('false');

% Input
if isempty(F), [n Ffit LL xest yfit] = deal([],[],0,[],[]); return, end
if ~isvector(F), error argument, end

% Physiological parameters
if length(par.F0)~=2, error 'F0 should be an interval', end
F0 = mean(par.F0);
baselineinterval = par.F0 / F0;
a = par.a;
decay = exp(-par.dt/par.tau);
sat = par.saturation;
if ~isempty(par.pnonlin) || par.hill~=1, error 'nonlinearity not implemented yet', end
spikerate = par.finetune.spikerate;
if isempty(par.finetune.sigma), error programming, end
sigmay = par.finetune.sigma/F0;

% Algo parameters
estimate = par.algo.estimate;
doMAP = strcmpi(estimate,'MAP');
doproba = strcmp(estimate,'proba');
dosample = ismember(estimate,{'sample' 'samples'});
interpmode = fn_switch(doMAP,par.algo.interpmode,'linear'); % spline interpolation can yield negative weights, which is not acceptable for probabilities
nsample = fn_switch(dosample,par.algo.nsample,1);

% Special: non-integer spikes
if par.special.nonintegerspike_minamp, error 'noninteger spikes not handled with a fixed baseline', end

% Get the normalized fluorescence
y = F/F0;
y = double(y(:));

% Algorithm parameters
nc = par.algo.nc; if isempty(nc), nc = par.algo.nc_norise; end % discretization
cmax = par.algo.cmax;
dc = cmax/(nc-1);
cc = (0:nc-1)'*dc; % column vector
nb = par.algo.nb; 
if isempty(nb)
    nb = par.algo.nb_nodrift;
end
db = diff(baselineinterval)/(nb-1);
bb = linspace(baselineinterval(1),baselineinterval(2),nb); % row vector

% Sizes
T = length(y);

% Precomputations for the interpolation function x <- x*decay + n
% (value before to look at if there was 0, 1, 2, 3 spikes)
nspikemax = 3;
cc0 = cc*decay;
cc1 = min(cc0 + 1,cmax);
cc2 = min(cc1 + 1,cmax);
cc3 = min(cc2 + 1,cmax);
% (interpolation matrices)
M0 = interp1(cc,eye(nc),cc0,interpmode);
M1 = interp1(cc,eye(nc),cc1,interpmode);
M2 = interp1(cc,eye(nc),cc2,interpmode);
M3 = interp1(cc,eye(nc),cc3,interpmode);
if doMAP
    MM = [M0; M1; M2; M3];
    MM = sparse(MM);
end

% Precomputations for the spike likelihood
% p(n) = exp(-rate*dt) (rate*dt)^n/n!, then take the negative log
nspikmax = 3;
if spikerate
    if par.special.burstcostsone
        nspikcost = [0 ones(1,nspikmax)];
    else
        nspikcost = 0:nspikmax;
    end
    lspike = +spikerate*par.dt +log(factorial(nspikcost)) -nspikcost*log(spikerate*par.dt);
    pspike = exp(-lspike)/sum(exp(-lspike)); % make the sum 1
    lspike = -log(pspike);
else
    % no a priori on spikes!
    lspike = zeros(1,1+nspikmax);
end

% Precomputation for probability update
% f1(c) = sum_n p(n) f(c*decay+n)
if ~doMAP
    MS = pspike(1)*M0 + pspike(2)*M1 + pspike(3)*M2 + pspike(4)*M3;
end

% Precomputation for the measure after saturation
% p(y|x) = 1/(sqrt(2*pi)*sigma) exp(-(y-a*x/(1+sat*x))^2/2*sigma^2)
dye = 1 + a * cc./(1+sat*cc);
xxmeasure = fn_mult(dye,bb);
lmeasure = -log(1/(sqrt(2*pi)*sigmay));

% Precomputation for the a priori probability of calcium c(1)
% m = spikerate*par.dt/(1-decay);
% v = spikerate*par.dt/(1-decay^2);
% pcalcium = 1/(sqrt(2*pi*v))*exp(-(cc-m).^2/(2*v));
% pcalcium = pcalcium / sum(pcalcium); % re-normalize
% lcalcium = -log(pcalcium);
% lcalcium = repmat(lcalcium,[1 nb]);
lcalcium = zeros(nc,nb);

% Debug display
if DEBUG && doMAP
    tt = (0:T-1)*par.dt;
    figure(429), clf
    hda=subplot(321); hdb=subplot(322);
    hdc=subplot(323); hdd=subplot(324); 
    hde=subplot(325); hdf=subplot(326);
    plot(tt,y,'parent',hda)
    hx = line(0,0,'linestyle','none','marker','*','color','k','parent',hda);
end

% Backward collecting/sampling/smoothing sweep
% L(c,b,t) remembers what is the best log-likelihood with ct=c and constant baseline=b
% L(c,b,t) = min_{n(t+1),..,n(T)} -log(p(c(t+1),..,c(T),y(t),..,y(T)|c(t)=c,all b(t')=b))
% while N(x,t+1) remembers the number of spikes between t and t+1
% N(c,b,t) = argmin_n(t+1) min_{n(t+2),..,n(T)} -log(p(c(t+1),..,c(T),y(t+1),..,y(T)|c(t)=c,all b(t')=b))
if ~doMAP, L = zeros(nc,nb,T); end
N = zeros(nc,nb,T,'uint8');
% fn_progress('backward',T)
for t=T:-1:1
%     fn_progress(t)
    % L(c,b,t) = min_n(t+1) -log(p(n(t+1)) + L(c(t+1),b,t+1)   <- time update 
    %             - log(p(y(t)|x(t))                           <- measure update
        
    if DEBUG && doMAP, set(hx,'xdata',tt(t),'ydata',y(t)), end
    
    % Time update (find the best n(t+1))
    if t==T
        % initialization with 'empty probability' p([])
        lt = zeros(nc,nb);
    else
        if doMAP
            % each column in 'lt1' corresponds to a different number of
            % spikes, then we find what is the optimal number of spikes that
            % gives the minimum
            lt1 = fn_add(lspike, reshape(MM*lt,nc,nspikemax+1,nb));
            [lt n1] = min(lt1,[],2);
            lt = squeeze(lt); n1 = squeeze(n1-1);
            N(:,:,t+1) = n1;

            if DEBUG
                imagesc(cc,bb,log2proba(lt)','parent',hde,[0 1e-3])
                imagesc(cc,bb,N(:,:,t+1)','parent',hdf,[0 3])
                set([hde hdf],'ydir','normal')
            end
        else
            lt = logmultexp(MS,lt); % interpolation of probabilities rather than of log-probabilities
        end
    end
    
    % Measure update
    lt = lt + (lmeasure+(y(t)-xxmeasure).^2/(2*sigmay^2));
    if ~doMAP, L(:,:,t) = lt; end
    
    % A priori on calcium concentration at t=1
    if t==1
        lt = lt + lcalcium;
    end
    
    if DEBUG && doMAP
        imagesc(cc,bb,log2proba(lt)','parent',hdb,[0 1e-3])
        set(hdb,'ydir','normal')
        drawnow
    end
end

% Precomputations for forward sweep
if doproba
    % Precomputations for the interpolation function f1(c) = f((c-n)/decay)
    % (value before to look at if there was 0, 1, 2, 3 spikes)
    cc0 = cc/decay;
    cc1 = (cc-1)/decay;
    cc2 = (cc-2)/decay;
    cc3 = (cc-3)/decay;
    % (interpolation matrices: interpolated probabilities will be zero where cc<nspike)
    M0 = interp1(cc,eye(nc),cc0,interpmode,0);
    M1 = interp1(cc,eye(nc),cc1,interpmode,0);
    M2 = interp1(cc,eye(nc),cc2,interpmode,0);
    M3 = interp1(cc,eye(nc),cc3,interpmode,0);
    
    % Precomputation for probability update
    % f1(c) = sum_n p(n) f((c-n)/decay)
    MS = pspike(1)*M0 + pspike(2)*M1 + pspike(3)*M2 + pspike(4)*M3;
    % f1(c) = sum_n n p(n) f((c-n)/decay)
    NS = pspike(2)*M1 + 2*pspike(3)*M2 + 3*pspike(4)*M3;
end

% Minimization to find the best baseline value (MAP estimations only!)
if doMAP
    % here lt becomes a vector
    [dum cbidxmin] = min(lt(:)); %#ok<*ASGLU>
    bidxmin = ceil(cbidxmin/nc);
    baseline = bb(bidxmin);
    lt = lt(:,bidxmin);
    N = squeeze(N(:,bidxmin,:));
elseif doproba
    % here lt remains a matrix!
    pt = exp(min(lt(:))-lt);
    pbaseline = sum(pt); % marginalize over calcium to get p(baseline|y)
    pbaseline = pbaseline/sum(pbaseline); % make sum 1
    baseline = sum(bb.*pbaseline); % E(baseline|y)
end

% forward sweep
n = zeros(T,nsample);
xest = zeros(T,nsample);
decay = exp(-par.dt/par.tau);
for t=1:T
	% calcium evolution
    if t==1
        if doMAP
            [LL cidx] = min(lt); % LL is the minimum negative log likelihood
            xest(t) = cc(cidx);
        elseif dosample
            % initiate samples
            LL = []; % would be quite useless to compute log likelihoods, isn't it?
            [cidx bidx] = logsample(lt,nsample);
            xest(t,:) = cc(cidx);
            baseline = bb(bidx);
        elseif doproba
            % integral
            LL = logsumexp(lt);
            xest(t) = sum(row(fn_mult(cc,log2proba(lt))));
        end
    else
        if doMAP
            n(t) = N(cidx,t);
            xest(t) = min(xest(t-1)*decay + n(t),cc(end));
            cidx = 1+round(xest(t)/dc);
        elseif dosample
            nspike = 0:nspikmax;                            % putative number of spikes
            ct = fn_add(column(xest(t-1,:))*decay,nspike);  % corresponding putative calcium values
            Lt = L(:,:,t);                                  % -log p(yt,..,yT|xt) [size nc*nb]
            if nb==1
                Lt = interp1(Lt,1+ct/dc,'linear',Inf);      % same, interpolated to the putative state values [size nsample*(1+nspikmax)]
            else
                Lt = interpn(Lt,1+ct/dc,repmat(bidx(:),1,1+nspikmax),'linear',Inf); % same, interpolated to the putative state values [size nsample*(1+nspikmax)]
            end
            lt = fn_add(lspike,Lt);                         % ~ -log p(xt|x(t-1),yt,..,yT)
            cidx = logsample(lt,'rows');     % selected number of spike
            n(t,:) = cidx-1;
            idx = sub2ind([nsample 1+nspikmax],1:nsample,row(cidx));
            xest(t,:) = ct(idx);
        elseif doproba
            % time update
            % for the moment:
            % . lt is -log p(x(t-1)|y1,..,y(t-1))
            % . L(:,t) is -log p(yt,..,yT|xt)
            % complicate ways of computing are needed to avoid numerical
            % errors
            % operations are performed columnwise (i.e. operating on
            % calcium, but performed independently for each baseline value;
            % in theory the marginal probability of baseline should remain
            % the same, but numerical errors would departure from it)
            lt1 = lt;                           % -log p(c(t-1)|b,y1,..,y(t-1))
            lmin = min(lt1);
            pt1 = exp(fn_subtract(lmin,lt1));   % ~ p(c(t-1)|b,y1,..,y(t-1))
            pt = MS*pt1;                        % ~ p(ct|b,y1,..,y(t-1))
            nt = (NS*pt1)./pt; nt(pt==0) = 0;   % E(nt|ct,b,y1,..,y(t-1))
            lt = fn_subtract(lmin,log(pt));     % -log p(ct|b,y1,..,y(t-1))
            lty = lt + L(:,:,t);                % ~ -log p(ct|b,y)
            pty = exp(fn_subtract(min(lty),lty));   % ~ p(ct|b,y)
            pty = fn_div(pty,sum(pty));         % make sum 1 in each column
            pty = fn_mult(pty,pbaseline);       % ~ p(ct,b|y), global sum is 1
            n(t) = sum(row(nt.*pty));           % E(nt|y)
            xest(t) = sum(row(fn_mult(cc,pty)));% E(xt|y)
            
            % measure update
            lt = lt + (lmeasure+(y(t)-xxmeasure).^2/(2*sigmay^2));
        end
    end

end


% Graph summary
if par.dographsummary
    % init graphics
    ha = initgraphsummary();
    tt = (0:T-1)*par.dt;
    % calcium
    imagesc(tt,cc,80+2*(-1).^(1:nc)'*ones(1,T),'parent',ha(1),[0 100])
    line(tt,mean(xest,2),'parent',ha(1))
    ylabel(ha(1),'calcium')
    %     set(ha(1),'ylim',[min(xest(:,1))-2*dc max(xest(:,1))+2*dc])
    % baseline
    imagesc(tt,bb*F0,80+2*(-1).^(1:nb)'*ones(1,T),'parent',ha(2),[0 100])
    line(tt([1 end]),mean(baseline)*F0*[1 1],'parent',ha(2))
    xlabel(ha(2),'time (s)')
    ylabel(ha(2),'baseline')
    %     set(ha(2),'ylim',[min(xest(:,2))-2*db max(xest(:,2))+2*db])
    set(ha,'ydir','normal')
    % clear third graph
    cla(ha(3))
    set(ha(3),'xtick',[],'ytick',[],'box','on')
    drawnow
end

% Saturation and scaling
xsaturation = a * xest./(sat*xest+1);

% Predicted measure (taking drifts into account)
yfit = fn_mult(1+xsaturation,row(baseline));

% Back from normalized to data scale
Ffit = yfit*F0;

% Change F0 to the actual baseline average
F0 = baseline*F0;

%-------------------------------------------------------------------------%
% baseline drift
function [n Ffit F0 LL xest yfit]= maxposterior_driftstate(F,par)

DEBUG = eval('false');

% Input
if isempty(F), [n Ffit LL xest yfit] = deal([],[],0,[],[]); return, end
if ~isvector(F), error argument, end

% Physiological parameters
if length(par.F0)~=2, error 'when estimating a drift, F0 should be an interval', end
switch par.drift.effect
    case 'additive'
        F0 = 1;
        if ~(F0>par.F0(1) && F0<par.F0(2))
            error 'additive drifts: calcium signals must be already normalized by F0'
        else
            disp 'additive drifts: calcium signals supposed already normalized by F0'
        end
    case 'multiplicative'
        F0 = mean(par.F0);
end
baselineinterval = par.F0 / F0;
a = par.a;
decay = exp(-par.dt/par.tau);
sat = par.saturation;
pnonlin = par.pnonlin;
if ~isempty(pnonlin) && sat~=0, error 'saturation and nonlinearity cannot be applied simultaneously', end
hill = par.hill;
spikerate = par.finetune.spikerate;
if isempty(par.finetune.sigma), error programming, end
sigmay = par.finetune.sigma/F0;
sigmab = par.drift.parameter/F0 * sqrt(par.dt);
if sigmab==0, error programming, end

% Algo parameters
estimate = par.algo.estimate;
doMAP = strcmpi(estimate,'MAP');
doproba = strcmp(estimate,'proba');
dosample = ismember(estimate,{'sample' 'samples'});
interpmode = fn_switch(doMAP,par.algo.interpmode,'linear'); % spline interpolation can yield negative weights, which is not acceptable for probabilities
nsample = fn_switch(dosample,par.algo.nsample,1);

% Special: non-integer spikes
nonintegerspike = par.special.nonintegerspike_minamp;
if nonintegerspike && ~doMAP, error 'noninteger spike are available only for MAP estimations', end

% Get the normalized fluorescence
y = F/F0;
y = double(y(:));

% Algorithm parameters
nc = par.algo.nc; if isempty(nc), nc = par.algo.nc_norise; end % discretization
cmax = par.algo.cmax;
dc = cmax/(nc-1);
cc = (0:nc-1)'*dc; % column vector
nb = par.algo.nb; if isempty(nb), nb = par.algo.nb_driftstate; end
db = diff(baselineinterval)/(nb-1);
bb = linspace(baselineinterval(1),baselineinterval(2),nb); % row vector

% Sizes
T = length(y);

% Precomputations for the interpolation function x <- x*decay + n
% (value before to look at if there was 0, 1, 2, 3 spikes)
nspikemax = 3;
cc0 = cc*decay;
cc1 = min(cc0 + 1,cmax);
cc2 = min(cc0 + 2,cmax);
cc3 = min(cc0 + 3,cmax);
% (interpolation matrices)
M0 = interp1(cc,eye(nc),cc0,interpmode);
if nonintegerspike==0
    M1 = interp1(cc,eye(nc),cc1,interpmode);
    M2 = interp1(cc,eye(nc),cc2,interpmode);
    M3 = interp1(cc,eye(nc),cc3,interpmode);
    if doMAP
        MM = [M0; M1; M2; M3];
        MM = sparse(MM);
    end
else
    M0 = sparse(M0);
    minjump = ceil(nonintegerspike/dc); % minimal calcium jump of an event
end

% Precomputations for the spike likelihood
% p(n) = exp(-rate*dt) (rate*dt)^n/n!, then take the negative log
nspikmax = 3;
if spikerate
    if par.special.burstcostsone
        nspikcost = [0 ones(1,nspikmax)];
    else
        nspikcost = 0:nspikmax;
    end
    lspike = +spikerate*par.dt +log(factorial(nspikcost)) -nspikcost*log(spikerate*par.dt);
    pspike = exp(-lspike)/sum(exp(-lspike)); % make the sum 1
    lspike = -log(pspike);
else
    % no a priori on spikes!
    lspike = zeros(1,1+nspikmax);
end

% Precomputation for probability update
% f1(c) = sum_n p(n) f(c*decay+n)
if ~doMAP
    MS = pspike(1)*M0 + pspike(2)*M1 + pspike(3)*M2 + pspike(4)*M3;
end

% Precomputation for the baseline drift
if doMAP
    % time update will involve finding the baseline drift that maximizes
    % probability; the maximum on the discretization grid will first be
    % located, then interpolation will be used to find the maximum with a
    % finer resolution
    % (drifting matrix)
    maxdrift = max(2,ceil(3*sigmab/db));
    DD = zeros(nb,2*maxdrift+1);
    for i=1:2*maxdrift+1
        DD(:,i) = max(min((1:nb)+(i-1-maxdrift),nb),1);
    end
    % ldrift = -log(1/(sqrt(2*pi)*sigmab)) + ((-maxdrift:maxdrift)*db).^2/(2*sigmab^2);
    ldrift = ((-maxdrift:maxdrift)*db).^2/(2*sigmab^2);
    ldrift = repmat(shiftdim(ldrift,-1),[nc nb]);
    % (quadratic interpolation of a triplet of points)
    tmp = eye(3);
    QQ = zeros(3);
    for i=1:3, QQ(:,i) = polyfit([-1 0 1],tmp(i,:),2); end
    QQ = QQ'; % operation on columns
else
    % time update will involve averaging probabilities accross possible
    % baseline drifts, which are described by a continuous (rather than
    % discrete) probability
    % we can construct a matrix multiplication that will realize the
    % interpolation and averaging at once; this matrix is obtained by first
    % replacing the continuous distribution by a fine-grain discrete
    % distribution
    discretesteps = (-6:.05:6); % to be multiplied with sigmab
    ndrift = length(discretesteps);
    pdrift = exp(-discretesteps.^2/2);
    pdrift = pdrift/sum(pdrift);
    bb1 = fn_add((1:nb)',discretesteps*(sigmab/db));
    BB = interp1(eye(nb),bb1(:),'linear',NaN); % (nb*ndrift)*nb
    BB = reshape(BB,[nb ndrift nb]);
    pdriftc = repmat(pdrift,[nb 1 nb]);
    pdriftc(isnan(BB)) = 0; pdriftc = fn_div(pdriftc,sum(pdriftc,2));
    BB(isnan(BB)) = 0;
    BB = squeeze(sum(BB.*pdriftc,2)); % nb*nb  
    BB = BB'; % will operate on columns
end

% Precomputation for the measure after saturation/nonlinearity
ccn = (par.c0+cc).^hill-par.c0^hill;
if isempty(pnonlin)
    % p(y|x) = 1/(sqrt(2*pi)*sigma) exp(-(y-a*x/(1+sat*x))^2/2*sigma^2)
    dye = 1 + a * ccn./(1+sat*ccn);
else
    dye = 1 + a * polyval([fliplr(pnonlin) 1-sum(pnonlin) 0],ccn);
end
switch par.drift.effect
    case 'additive'
        xxmeasure = fn_add(dye-1,bb);
    case 'multiplicative'
        xxmeasure = fn_mult(dye,bb);
    otherwise
        error flag
end
lmeasure = -log(1/(sqrt(2*pi)*sigmay));

% Precomputation for the a priori probability of calcium c(1)
% m = spikerate*par.dt/(1-decay);
% v = spikerate*par.dt/(1-decay^2);
% pcalcium = 1/(sqrt(2*pi*v))*exp(-(cc-m).^2/(2*v));
% pcalcium = pcalcium / sum(pcalcium); % re-normalize
% lcalcium = -log(pcalcium);
% lcalcium = repmat(lcalcium,[1 nb]);
lcalcium = zeros(nc,nb);

% Debug display
if DEBUG
    tt = (0:T-1)*par.dt;
    figure(429), clf
    hda=subplot(321); 
    plot(tt,y*F0,'parent',hda)
    hx = line(0,0,'linestyle','none','marker','*','color','k','parent',hda);
    hdb=subplot(322);
    hdc=subplot(323); hdd=subplot(324); 
    hde=subplot(325); 
    ime = imagesc(cc,bb*F0,zeros(nb,nc),'parent',hde,[0 1e-3]);
    xlabel(hde,'calcium'), ylabel(hde,'baseline'), set(hde,'ydir','normal')
    hdf=subplot(326);
end

% Backward sweep
% L(x,t) remembers what is the best log-likelihood with xt=x
% L(x,t) = min_{x(t+1),..,x(T)} -log(p(x(t+1),..,x(T),y(t),..,y(T)|x(t)=x))
% while N(x,t+1) and D(x,t+1) remember respectively the number of spikes
% between t and t+1 and the baseline drift that give this best likelihood
% N(c,b,t) = argmin_n(t+1) min_{x(t+2),..,x(T)}        -log(p(n(t+1),x(t+2),..,x(T),y(t+1),..,y(T)|c(t)=c,b(t+1)=b))
% D(c,b,t) = argmin_b(t+1) min_{n(t+1),x(t+2),..,x(T)} -log(p(x(t+1),x(t+2),..,x(T),y(t+1),..,y(T)|c(t)=c,b(t)=b))
if ~doMAP, L = zeros(nc,nb,T); end
if doMAP
    D = zeros(nc,nb,T,'single');
    if nonintegerspike==0
        N = zeros(nc,nb,T,'uint8');
    else
        N = zeros(nc,nb,T,'single');
    end
end
for t=T:-1:1
    % L(x,t) = min_n(t+1) -log(p(n(t+1)) + min_b(t+1) -log(p(b(t+1)|b(t))) + L(x(t+1),t+1)   <- time update (minimize first over the drift in b, then over the number of spikes)
    %          - log(p(y(t)|x(t))                                                            <- measure update
    
    if DEBUG, set(hx,'xdata',tt(t),'ydata',y(t)*F0), end
    
    % Time update (find the best n(t+1))
    if t==T
        % initialization with 'empty probability' p([])
        lt = zeros(nc,nb);
    else
        % calcium time update
        if doMAP && ~nonintegerspike
            % what is the best number of spikes
            %             lt1 = lspike_interp + reshape(MM*lt,nc,nspikemax+1,nb);
            lt1 = fn_add(lspike, reshape(MM*lt,nc,nspikemax+1,nb));
            [lt n1] = min(lt1,[],2);
            lt = squeeze(lt); n1 = squeeze(n1-1);
            N(:,:,t+1) = n1;
        elseif doMAP && nonintegerspike
            % decay
            lt1_noevent = lspike(1)+M0*lt; % lspike(1) is the cost of zero spike
            % jump
            lt1_event = lt1_noevent;
            jumps = zeros(nc,nb);
            % (initialize with hypothetical jumps (nc-minevent)->nc+1 of
            % infinite cost)
            ltk = Inf;
            jumpk = minjump+1;
            for k=nc:-1:(1+minjump)
                ltk1 = lt(k,:);
                smaller = (ltk1<=ltk); % does a jump of amplitude minjump reach a value which is more interesting than the current minimum?
                ltk(smaller) = ltk1(smaller);
                jumpk(smaller) = minjump;
                jumpk(~smaller) = jumpk(~smaller)+1;
                lt1_event(k-minjump,:) = ltk;
                jumps(k-minjump,:) = jumpk;
            end
            lt1_event = lspike(2)+lt1_event; % lspike(2) is the cost of 1 spike (here, 1 'event') 
            % what is better between the decay and the optimal jump
            lt = lt1_noevent;
            smaller = (lt1_event<lt1_noevent);
            lt(smaller) = lt1_event(smaller);
            N = reshape(N,[nc*nb T]);
            N(smaller,t) = jumps(smaller)*dc;
            N = reshape(N,[nc nb T]);
        elseif ~doMAP
            lt = logmultexp(MS,lt); % interpolation of probabilities rather than of log-probabilities
        end
        
        if DEBUG
            set(ime,'cdata',log2proba(lt)')
            set(hde,'ydir','normal')
            if doMAP
                imagesc(cc,bb,N(:,:,t+1)','parent',hdf,[0 3])
                set(hdf,'ydir','normal')
            end
            drawnow
        end
        
        % baseline time update
        if doMAP
            % what is the optimal baseline drift
            % get: lt = min_b(t+1) -log(p(b(t+1)|b(t))) + L(x(t+1),t+1)
            lt1 = ldrift + reshape(lt(:,DD),[nc nb 2*maxdrift+1]);
            [lt idrift] = min(lt1,[],3);
            
            % find a over-sampling minimum using a quadratic interpolation when
            % drifting values are not on the sides defined by the maximum
            % allowed
            oksides = ~(idrift==1 | idrift==2*maxdrift+1);
            lt1 = reshape(lt1,[nc*nb 2*maxdrift+1]);
            lt1ok = lt1(oksides,:);
            idriftok = idrift(oksides);
            nok = sum(oksides(:));
            indices3 = fn_add((1:nok)'+nok*(idriftok-1),nok*[-1 0 1]);
            values3 = lt1ok(indices3);
            qq = values3 * QQ;
            idriftmin = -qq(:,2)./(2*qq(:,1)); % (q(x) = ax^2 + bx + c -> the min is -b/2a)
            idrift(oksides) = idrift(oksides) + idriftmin;
            lt(oksides) = (qq(:,1).*idriftmin + qq(:,2)).*idriftmin + qq(:,3);
            
            D(:,:,t+1) = (idrift-1-maxdrift)*db;
            
            if DEBUG
                imagesc(cc,bb,log2proba(lt)','parent',hdc,[0 1e-3])
                imagesc(cc,bb,D(:,:,t+1)','parent',hdd,[-1 1]*maxdrift*db)
                set([hdc hdd],'ydir','normal')
            end
        else
            lt = logmultexp_column(BB,lt); % interpolation of probabilities
        end
    end
    
    % Measure update
    lt = lt + (lmeasure+(y(t)-xxmeasure).^2/(2*sigmay^2));
    if ~doMAP, L(:,:,t) = lt; end
    
    % A priori on calcium concentration at t=1
    if t==1
        lt = lt + lcalcium;
    end
    
    if DEBUG && doMAP
        imagesc(cc,bb,log2proba(lt)','parent',hdb,[0 1e-3])
        set(hdb,'ydir','normal')
        drawnow
    end
end

% Precomputations for forward sweep
if doproba
    % Precomputations for the interpolation function f1(c) = f((c-n)/decay)
    % (value before to look at if there was 0, 1, 2, 3 spikes)
    cc0 = cc/decay;
    cc1 = (cc-1)/decay;
    cc2 = (cc-2)/decay;
    cc3 = (cc-3)/decay;
    % (interpolation matrices: interpolated probabilities will be zero where cc<nspike)
    M0 = interp1(cc,eye(nc),cc0,interpmode,0);
    M1 = interp1(cc,eye(nc),cc1,interpmode,0);
    M2 = interp1(cc,eye(nc),cc2,interpmode,0);
    M3 = interp1(cc,eye(nc),cc3,interpmode,0);
    
    % Precomputation for probability update
    % f1(c) = sum_n p(n) f((c-n)/decay)
    MS = pspike(1)*M0 + pspike(2)*M1 + pspike(3)*M2 + pspike(4)*M3;
    % f1(c) = sum_n n p(n) f((c-n)/decay)
    NS = pspike(2)*M1 + 2*pspike(3)*M2 + 3*pspike(4)*M3;
elseif dosample
    %discretesteps = [-6 -5:.5:-2.5 -2:.1:-.1 -.05 0 .05 .1:.1:2 2.5:.5:5 6]; % try a limited number of drifts!
    %discretesteps = -6:.05:6;
    %pdrift = exp(-discretesteps.^2/2);
    %pdrift = pdrift/sum(pdrift);
    ldrift = -log(pdrift);
    lspike_drift = fn_add(column(lspike),row(ldrift));
end

% Forward collecting/sampling/smoothing step
n = zeros(T,nsample);
xest = zeros(T,2,nsample);
if dosample, fn_progress('sampling',T), end
for t=1:T
    if dosample, fn_progress(t), end
    if t==1
        if doMAP
            if par.drift.baselinestart
                % impose that the initial calcium level is baseline
                cidx = 1;
                ystart = mean(y(1:ceil(0.1/par.dt))); % average over 100ms to get the start value
                [dum bidx] = min(abs(ystart-xxmeasure(cidx,:))); %#ok<ASGLU>
                LL = lt(cidx,bidx);
            else
                % LL is the minimum negative log likelihood
                [LL cidx] = min(lt,[],1);
                [LL bidx] = min(LL,[],2);
                cidx = cidx(bidx);
            end
            xest(t,:) = [cc(cidx) bb(bidx)];
        elseif dosample
            % initiate samples
            LL = []; % would be quite useless to compute log likelihoods, isn't it?
            [cidx bidx] = logsample(lt,nsample);
            xest(t,1,:) = cc(cidx);
            xest(t,2,:) = bb(bidx);
        elseif doproba
            LL = logsumexp(lt(:));
            pt = log2proba(lt);
            xest(t,1) = sum(row(fn_mult(cc,pt)));
            xest(t,2) = sum(row(fn_mult(bb,pt)));
        end
    else
        if doMAP
            xest(t,2) = fn_coerce(xest(t-1,2) + D(cidx,bidx,t),baselineinterval);
            bidx = 1+round((xest(t,2)-bb(1))/db);
            n(t) = N(cidx,bidx,t);
            xest(t,1) = min(xest(t-1,1)*decay + n(t),cmax);
            cidx = 1+round(xest(t,1)/dc);
        elseif dosample
            % draw calcium and baseline evolutions at once
            % too difficult this time to do all particles at once 
            % -> use a for loop
            nspike = column(0:nspikmax);                    % putative number of spikes
            Lt = L(:,:,t);                                  % -log p(yt,..,yT|xt) [size nc*nb]
            for ksample = 1:nsample
                ct = xest(t-1,1,ksample)*decay + nspike;    % corresponding putative calcium values
                bt = xest(t-1,2,ksample) + discretesteps*sigmab;        % putative baseline values
                ltk0 = interpn(Lt,1+ct/dc,1+(bt-bb(1))/db,'linear',Inf); % -log p(yt,..,yT|ct,Bt) [size (1+nspikmax)*nb]
                ltk = lspike_drift + ltk0;                   % ~ -log p(xt|x(t-1),yt,..,yT) [size (1+nspikmax)*ndrift]
                [cidx bidx] = logsample(ltk);
                n(t,ksample) = cidx-1;
                xest(t,1,ksample) = ct(cidx);
                xest(t,2,ksample) = bt(bidx);
            end
        elseif doproba
            % time update
            % for the moment:
            % . lt is -log p(x(t-1)|y1,..,y(t-1))
            % . L(:,:,t) is -log p(yt,..,yT|xt)
            % complicate ways of computing are needed to avoid numerical
            % errors
            lt1 = lt;               % -log p(x(t-1)|y1,..,y(t-1))
            lmin = min(lt1(:));
            pt1 = exp(lmin-lt1);    % ~ p(x(t-1)|y1,..,y(t-1))
            pt = MS*pt1*BB;         % ~ p(xt|y1,..,y(t-1))
            nt = (NS*pt1*BB)./pt; nt(pt==0) = 0;   % E(nt|xt,b(t-1),y1,..,y(t-1))
            lt = lmin-log(pt);      % -log p(xt|y1,..,y(t-1))
            lty = lt + L(:,:,t);    % ~ -log p(xt|y)
            L(:,:,t) = lty;
            pty = exp(min(lty(:))-lty);             % ~ p(xt|y)
            pty = pty/sum(pty(:));
            n(t) = sum(row(nt.*pty));               % E(nt|y)
            xest(t,1) = sum(row(fn_mult(cc,pty)));  % E(ct|y)
            xest(t,2) = sum(row(fn_mult(bb,pty)));  % E(bt|y)
            
            % measure update
            lt = lt + (lmeasure+(y(t)-xxmeasure).^2/(2*sigmay^2));
        end
    end
end

% Graph summary
if par.dographsummary
    % probabilities (MAP and samples: from future only; proba: full posterior)
    showproba = doproba;
    if showproba
        P = L; for t=1:T, P(:,:,t) = log2proba(L(:,:,t)); end
        PC = fn_normalize(sum(P,2),1,'proba');
        PB = fn_normalize(sum(P,1),2,'proba');
    end
    % init graphics
    ha = initgraphsummary();
    tt = (0:T-1)*par.dt;
    % calcium & baseline
    im = .8+.02*(-1).^(1:nc)'*ones(1,T);
    if showproba
        blue = squeeze(PC);
        im = repmat(im.*(1-blue),[1 1 3]);
        im(:,:,3) = im(:,:,3)+blue;
        imagesc(tt,cc,im,'parent',ha(1))
    else
        imagesc(tt,cc,im,'parent',ha(1),[0 1])
        line(tt,squeeze(xest(:,1,:)),'parent',ha(1))
    end
    ylabel(ha(1),'calcium')
    %     set(ha(1),'ylim',[min(xest(:,1))-2*dc max(xest(:,1))+2*dc])
    im = .8+.02*(-1).^(1:nb)'*ones(1,T);
    if showproba
        blue = squeeze(PB);
        im = repmat(im.*(1-blue),[1 1 3]);
        im(:,:,3) = im(:,:,3)+blue;
        imagesc(tt,bb*F0,im,'parent',ha(2),[0 1])
    else
        imagesc(tt,bb*F0,im,'parent',ha(2),[0 1])
        line(tt,squeeze(xest(:,2,:))*F0,'parent',ha(2))
    end
    xlabel(ha(2),'time (s)')
    ylabel(ha(2),'baseline')
    %     set(ha(2),'ylim',[min(xest(:,2))-2*db max(xest(:,2))+2*db])
    set(ha,'ydir','normal')
    % clear third graph
    cla(ha(3))
    set(ha(3),'xtick',[],'ytick',[],'box','on')
    drawnow
end

% Saturation/nonlinearity and scaling
cestn = squeeze(par.c0+xest(:,1,:)).^hill-par.c0^hill;
if isempty(pnonlin)
    xsaturation = a * cestn./(1+sat*cestn);
else
    xsaturation = a * polyval([fliplr(pnonlin) 1-sum(pnonlin) 0],cestn);
end

% Predicted measure (taking drifts into account)
switch par.drift.effect
    case 'additive'
        yfit = xsaturation + squeeze(xest(:,2,:));
    case 'multiplicative'
        yfit = (1+xsaturation).*squeeze(xest(:,2,:));
    otherwise
        error flag
end

% Back from normalized to data scale
Ffit = yfit*F0;

% Reajust F0 and xest(:,2) to make the mean of xest(:,2) 1
avgb = mean(row(xest(:,2,:)));
F0 = F0 * avgb;
xest(:,2,:) = xest(:,2,:) / avgb;

%-------------------------------------------------------------------------%
% baseline drift and rise time
function [n Ffit F0 LL xest yfit]= maxposterior_driftstaterise(F,par)

DEBUG = false;

% Input
if isempty(F), [n Ffit LL xest yfit] = deal([],[],0,[],[]); return, end
if ~isvector(F), error argument, end

% Physiological parameters
if length(par.F0)~=2, error 'when estimating a drift, F0 should be an interval', end
switch par.drift.effect
    case 'additive'
        F0 = 1;
        if ~(F0>par.F0(1) && F0<par.F0(2))
            error 'additive drifts: calcium signals must be already normalized by F0'
        else
            disp 'additive drifts: calcium signals supposed already normalized by F0'
        end
    case 'multiplicative'
        F0 = mean(par.F0);
end
baselineinterval = par.F0 / F0;
dt = par.dt;
a = par.a;
decay = exp(-dt/par.tau);
ton = par.ton;
sat = par.saturation;
pnonlin = par.pnonlin;
if ~isempty(pnonlin) && sat~=0, error 'saturation and nonlinearity cannot be applied simultaneously', end
hill = par.hill;
c0 = par.c0;
if isempty(c0)
    if hill~=1, error programming, end
    c0 = 0;
end
spikerate = par.finetune.spikerate;
if isempty(par.finetune.sigma), error programming, end
sigmay = par.finetune.sigma/F0;
sigmab = par.drift.parameter/F0 * sqrt(dt);
if sigmab==0, error programming, end

% Algo parameters
doMAP = strcmpi(par.algo.estimate,'MAP');
if ~doMAP
    error 'only MAP estimations are supported with a rise time'
end

% Special: non-integer spikes
if par.special.nonintegerspike_minamp, error 'noninteger spikes not handled with rise time', end

% Get the normalized fluorescence
y = F/F0;
y = double(y(:));

% Discretization
% (calcium)
nc = par.algo.nc; if isempty(nc), nc = par.algo.nc_rise; end % discretization
cmax = par.algo.cmax;
dc = cmax/(nc-1);
cc = (0:nc-1)'*dc; % column vector

% (probe, i.e. bound indicator)
cmaxn = (c0+cmax)^hill - c0^hill;
pmax = cmaxn ./ (1+sat*cmaxn); % remains btw. 0 and 1/sat
np = par.algo.np; 
if fn_dodebug, disp 'non-regular probe spacing', end
dph = pmax^(1/hill)/(np-1);
pp = linspace(0,pmax^(1/hill),np).^hill;
pp(end) = pmax; % necessary to avoid that through numerical errors we have pmax^(1/hill)^hill slightly less than pmax, resulting in interpolation errors later

% (baseline)
nb = par.algo.nb; if isempty(nb), nb = par.algo.nb_driftrise; end
db = diff(baselineinterval)/(nb-1);
bb = linspace(baselineinterval(1),baselineinterval(2),nb); % row vector

% Sizes
T = length(y);

% Precomputations for the interpolation function x <- x*decay + n
% (value before to look at if there was 0, 1, 2, 3 spikes)
if fn_dodebug, disp 'nspikemax = 1!', end
nspikemax = 1;
cc0 = cc*decay;
cc1 = min(cc0 + 1,cmax);
% cc2 = min(cc0 + 2,cmax);
% cc3 = min(cc0 + 3,cmax);
% (interpolation matrices)
M0 = interp1(cc,eye(nc),cc0,par.algo.interpmode);
M1 = interp1(cc,eye(nc),cc1,par.algo.interpmode);
% M2 = interp1(cc,eye(nc),cc2,par.algo.interpmode);
% M3 = interp1(cc,eye(nc),cc3,par.algo.interpmode);
% MM = [M0; M1; M2; M3];
if doMAP
    MM = [M0; M1];
    MM = sparse(MM);
end

% Precomputations for the spike likelihood
% p(n) = exp(-rate*dt) (rate*dt)^n/n!, then take the negative log
if spikerate
    if par.special.burstcostsone
        nspikcost = [0 ones(1,nspikemax)];
    else
        nspikcost = 0:nspikemax;
    end
    lspike = +spikerate*dt +log(factorial(nspikcost)) -nspikcost*log(spikerate*dt);
    lspike = -log(exp(-lspike)/sum(exp(-lspike))); % re-normalize
else
    % no a priori on spikes!
    lspike = zeros(1,1+nspikemax);
end
lspike = repmat(lspike,[nc 1 np*nb]);

% Precomputation for the probe bounding to calcium
% (use matrices calcium x probe)
ccc = repmat(column(cc),[1 np]);
cn = (c0+ccc).^hill - c0^hill;
ptarget = cn ./ (1+sat*cn); % remains btw. 0 and 1/sat
pspeed = (1+sat*cn) / ton;
ppp = repmat(row(pp),[nc 1]);
% (probe value at t+1)
ppp1 = ptarget + (ppp-ptarget).*exp(-pspeed*dt); % remains btw. 0 and 1/sat
% (interpolation: first compute the interpolation for each calcium value,
% then rearrange)
PP = cell(1,nc); 
for ic=1:nc
    PP{ic} = interp1(ppp(ic,:),eye(np),ppp1(ic,:),'linear'); 
end
PP = blkdiag(PP{:}); % (np*nc)^2 square matrix
PP = reshape(permute(reshape(PP,[np nc np nc]),[2 1 4 3]),[nc*np nc*np]); % (nc*np)^2 squarematrix
PP = sparse(PP);

% Precomputation for the baseline drift
% (drifting matrix)
maxdrift = max(2,ceil(3*sigmab/db));
DD = zeros(nb,2*maxdrift+1);
for i=1:2*maxdrift+1
	DD(:,i) = max(min((1:nb)+(i-1-maxdrift),nb),1);
end
% ldrift = -log(1/(sqrt(2*pi)*sigmab)) + ((-maxdrift:maxdrift)*db).^2/(2*sigmab^2);
ldrift = ((-maxdrift:maxdrift)*db).^2/(2*sigmab^2);
ldrift = repmat(shiftdim(ldrift,-1),[nc*np nb]);
% (quadratic interpolation of a triplet of points)
tmp = eye(3);
QQ = zeros(3);
for i=1:3, QQ(:,i) = polyfit([-1 0 1],tmp(i,:),2); end
QQ = QQ'; % operation on columns

% Precomputation for the measure
xxmeasure = a*ppp;
switch par.drift.effect
    case 'additive'
        xxmeasure = fn_add(xxmeasure,third(bb));
    case 'multiplicative'
        xxmeasure = fn_mult(1+xxmeasure,third(bb));
    otherwise
        error flag
end
lmeasure = -log(1/(sqrt(2*pi)*sigmay));

% Precomputation for the a priori probability of calcium c(1)
% m = spikerate*dt/(1-decay);
% v = spikerate*dt/(1-decay^2);
% pcalcium = 1/(sqrt(2*pi*v))*exp(-(cc-m).^2/(2*v));
% pcalcium = pcalcium / sum(pcalcium); % re-normalize
% lcalcium = -log(pcalcium);
% lcalcium = repmat(lcalcium,[1 np nb]);
lcalcium = zeros(nc,np,nb);

% Debug display
if DEBUG && doMAP
    tt = (0:T-1)*dt;
    figure(429), clf
    hda=subplot(321); hdb=subplot(322);
    hdc=subplot(323); hdd=subplot(324); 
    hde=subplot(325); hdf=subplot(326);
    plot(tt,y,'parent',hda)
    hx = line(0,0,'linestyle','none','marker','*','color','k','parent',hda);
end

% Backward sweep
% L(x,t) remembers what is the best log-likelihood with xt=x
% L(x,t) = min_{x(t+1),..,x(T)} -log(p(x(t+1),..,x(T),y(t),..,y(T)|x(t)=x))
% while N(x,t+1) and D(x,t+1) remember respectively the number of spikes
% between t and t+1 and the baseline drift that give this best likelihood
% N(c,b,t) = argmin_n(t+1) min_{x(t+2),..,x(T)}        -log(p(n(t+1),x(t+2),..,x(T),y(t+1),..,y(T)|c(t)=c,b(t+1)=b))
% D(c,b,t) = argmin_b(t+1) min_{n(t+1),x(t+2),..,x(T)} -log(p(x(t+1),x(t+2),..,x(T),y(t+1),..,y(T)|c(t)=c,b(t)=b))

% L = zeros(nc,np,nb,T,'single');
D = zeros(nc,np,nb,T,'int8');
N = zeros(nc,np,nb,T,'uint8');
fn_progress('backward',T)
for t=T:-1:1
    fn_progress(t)
    % L(x,t) = min_n(t+1) -log(p(n(t+1)) + min_b(t+1) -log(p(b(t+1)|b(t))) + L(x(t+1),t+1)   <- time update (minimize first over the drift in b, then over the number of spikes)
    %          - log(p(y(t)|x(t))                                                            <- measure update
    
       
    if DEBUG && doMAP, set(hx,'xdata',tt(t),'ydata',y(t)), end
    
    % Time update (find the best n(t+1))
    if t==T
        % initialization with 'empty probability' p([])
        lt = zeros(nc,np,nb);
    else
        % We start with:
        % lt(x(t+1)) = L(x(t+1),t+1)
        % lt(c(t+1),p(t+1),b(t+1)) = L(c(t+1),p(t+1),b(t+1),t+1)
        
        % 1) what is the optimal baseline drift
        % -> we get:
        % lt(c(t+1),p(t+1),b(t)) = min_b(t+1) -log(p(b(t+1)|b(t)) +  L(c(t+1),p(t+1),b(t+1),t+1)
        lt = reshape(lt,[nc*np nb]);
        lt1 = ldrift + reshape(lt(:,DD),[nc*np nb 2*maxdrift+1]);
        [lt idrift] = min(lt1,[],3);
        
        % find a over-sampling minimum using a quadratic interpolation when
        % drifting values are not on the sides defined by the maximum
        % allowed 
        oksides = ~(idrift==1 | idrift==2*maxdrift+1);
        lt1 = reshape(lt1,[nc*np*nb 2*maxdrift+1]);
        lt1ok = lt1(oksides,:);
        idriftok = idrift(oksides);
        nok = sum(oksides(:));
        indices3 = fn_add((1:nok)'+nok*(idriftok-1),nok*[-1 0 1]);
        values3 = lt1ok(indices3);
        qq = values3 * QQ;
        idriftmin = -qq(:,2)./(2*qq(:,1)); % (q(x) = ax^2 + bx + c -> the min is -b/2a)
        idrift(oksides) = idrift(oksides) + idriftmin;
        lt(oksides) = (qq(:,1).*idriftmin + qq(:,2)).*idriftmin + qq(:,3);
        lt = reshape(lt,[nc np nb]);

        %D(:,:,:,t+1) = (reshape(idrift,[nc np nb])-1-maxdrift)*db;
        Dt1 = (reshape(idrift,[nc np nb])-1-maxdrift)*db;
        D(:,:,:,t+1) = Dt1*(127/maxdrift/db);
        if DEBUG && doMAP
            imagesc(cc,bb,log2proba(lt)','parent',hdc,[0 1e-3])
            imagesc(cc,bb,Dt1','parent',hdd,[-1 1]*maxdrift*db)
            set([hdc hdd],'ydir','normal')
        end
        
        % 2) probe follows a deterministic evolution
        % -> we get (by a simple interpolation):
        % lt(c(t+1),p(t),b(t)) = min_b(t+1) -log(p(b(t+1)|b(t)) +  L(c(t+1),p(t+1),b(t+1),t+1) - log(p(y(t)|x(t)) 
        % [note that here there is an approximation that does not fit what
        % is written in the paper, because p(t+1) is assumed to be a
        % function of p(t) and c(t+1), instead of a function of p(t) and
        % c(t) as it should]
        lt = reshape(lt,[nc*np nb]);
        lt = PP*lt;
        lt = reshape(lt,[nc np nb]);
        
        % 3) what is the best number of spikes
        % -> we get:
        % lt(c(t),p(t),b(t)) = min_n(t+1) -log(p(n(t+1)) + min_b(t+1) -log(p(b(t+1)|b(t)) +  L(c(t+1),p(t+1),b(t+1),t+1)
        lt = reshape(lt,[nc np*nb]);
        lt1 = lspike + reshape(MM*lt,nc,nspikemax+1,np*nb);
        [lt n1] = min(lt1,[],2);
        lt = reshape(lt,[nc np nb]);
        N(:,:,:,t+1) = reshape(n1-1,[nc np nb]);
   
        if DEBUG && doMAP
            imagesc(cc,bb,log2proba(lt)','parent',hde,[0 1e-3])
            imagesc(cc,bb,N(:,:,t+1)','parent',hdf,[0 3])
            set([hde hdf],'ydir','normal')
        end
        
    end
    
    % Measure update
    % -> we get:
    % lt(c(t),p(t),b(t)) = min_n(t+1) -log(p(n(t+1)) + min_b(t+1) -log(p(b(t+1)|b(t)) +  L(c(t+1),p(t+1),b(t+1),t+1)
    lt = lt + (lmeasure+(y(t)-xxmeasure).^2/(2*sigmay^2));
    
    % A priori on calcium concentration at t=1
    if t==1
        lt = lt + lcalcium;
    end
    
    %         L(:,:,:,t) = lt;
    if DEBUG && doMAP % problem: no more 2D but 3D
        imagesc(cc,bb,log2proba(lt)','parent',hdb,[0 1e-3])
        set(hdb,'ydir','normal')
        drawnow
    end
end
fn_progress end

% Forward collecting/sampling/smoothing step
n = zeros(T,1);
xest = zeros(T,3);
% idxs = zeros(T,3);
for t=1:T
    if t==1
        if par.drift.baselinestart
            % impose that the initial calcium level is baseline 
            cidx = 1; pidx = 1;
            ystart = mean(y(1:ceil(0.1/dt))); % average over 100ms to get the start value
            [dum bidx] = min(abs(ystart-xxmeasure(cidx,pidx,:))); %#ok<ASGLU>
            LL = lt(cidx,pidx,bidx);
        else
            % LL is the minimum negative log likelihood
            [LL idx] = min(lt(:));
            [cidx pidx bidx] = ind2sub([nc np nb],idx);
        end
        xest(t,:) = [cc(cidx) ppp(cidx,pidx) bb(bidx)];
    else
        % baseline evolution
        Dt = double(D(cidx,pidx,bidx,t))/(127/maxdrift/db);
        xest(t,3) = fn_coerce(xest(t-1,3) + Dt,baselineinterval);
        bidx = 1+round((xest(t,3)-bb(1))/db);
        % calcium evolution
        n(t) = N(cidx,pidx,bidx,t);
        xest(t,1) = min(xest(t-1,1)*decay + n(t),cmax);
        cidx = 1+round(xest(t,1)/dc);
        % probe evolution
        cn = (c0+xest(t,1))^hill-c0^hill;
        ptarget = cn/(1+sat*cn);
        pspeed = (1+sat*cn)/ton;
        xest(t,2) = ptarget + (xest(t-1,2)-ptarget)*exp(-pspeed*dt);
        pidx = 1+round(xest(t,2)^(1/hill)/dph);
        if cidx==1, pidx=1; end
        if pidx>np
            if fn_dodebug, keyboard, end
            pidx=np;
        end
    end
    %     idxs(t,:) = [cidx pidx bidx];
    %L(cidx,bidx,t) = nan;
    %D(cidx,bidx,t) = nan;
end

% Graph summary
if par.dographsummary
    % init graphics
    ha = initgraphsummary();
    tt = (0:T-1)*dt;
    % calcium
    imagesc(tt,cc,80+2*(-1).^(1:nc)'*ones(1,T),'parent',ha(1),[0 100])
    line(tt,xest(:,1),'parent',ha(1))
    ylabel(ha(1),'calcium')
    %     set(ha(1),'ylim',[min(xest(:,1))-2*dc max(xest(:,1))+2*dc])
    % probe
    imagesc(tt,pp.^(1/hill),80+2*(-1).^(1:np)'*ones(1,T),'parent',ha(2),[0 100])
    line(tt,xest(:,2).^(1/hill),'parent',ha(2))
    ylabel(ha(2),'probe^{1/hill}')
    % baseline
    imagesc(tt,bb*F0,80+2*(-1).^(1:nb)'*ones(1,T),'parent',ha(3),[0 100])
    line(tt,xest(:,3)*F0,'parent',ha(3))
    xlabel(ha(3),'time (s)')
    ylabel(ha(3),'baseline')
    %     set(ha(2),'ylim',[min(xest(:,2))-2*db max(xest(:,2))+2*db])
    set(ha,'ydir','normal')
    drawnow
end

% Predicted measure (taking drifts into account)
switch par.drift.effect
    case 'additive'
        yfit = a*xest(:,2) + xest(:,3);
    case 'multiplicative'
        yfit = (1+a*xest(:,2)).*xest(:,3);
    otherwise
        error flag
end

% Back from normalized to data scale
Ffit = yfit*F0;

% Reajust F0 and xest(:,3) to make the mean of xest(:,3) 1
F0 = F0 * mean(xest(:,3));
xest(:,3) = xest(:,3) / mean(xest(:,3));


%-------------------------------------------------------------------------%
%                       UTILITY FUNCTIONS                                 %
%-------------------------------------------------------------------------%

%--- (builddrifts is used only in forward simulations, not in estimations)
function drifts = builddrifts(arg,nt)

if isstruct(arg)
    if ~isfield(arg.drift,'method') || ~strcmp(arg.drift.method,'basis functions'), drifts = 0; return, end
    matrixonly = false;
    a = arg.drift.estimate;
    ndrift = length(a);
    if ndrift~=arg.drift.parameter, error 'drift values do not match the number of drifts', end
elseif isnumeric(arg)
    matrixonly = false;
    a = arg;
    ndrift = length(a);
elseif iscell(arg)
    matrixonly = true;
    ndrift = arg{1};
end
if ndrift==0, drifts = zeros(nt,1); return, end
if ~mod(ndrift,2), error 'number of drifts must be odd', end

drifts = linspace(-1,1,nt)';
if ndrift>1
    phase = linspace(0,2*pi,nt)';
    nsin = (ndrift-1)/2;
    drifts(1,1+2*nsin) = 0; % pre-allocate
    for k=1:nsin
        drifts(:,2*k) = sin(k*phase);
        drifts(:,2*k+1) = cos(k*phase);
    end
end

if ~matrixonly
    drifts = drifts * a(:);
end

%---
function p = log2proba(l,dim)

if nargin<2, dim = []; end

if isempty(dim)
    l = l - min(l(:));
else
    % proba in dimension dim
    % e.g. if dim=2, each row is a probability distribution
    l = fn_subtract(l, min(l,[],dim));
end
p = exp(-l);
p(isnan(p)) = 0;
if isempty(dim)
    p = p/sum(p(:));
else
    p = fn_div(p,sum(p,dim));
end

%---
function ha = initgraphsummary()

hf = 253;
if ~ishandle(hf)
    figure(hf)
    fn_setfigsize(hf,400,800)
    set(hf,'numbertitle','off','name','tps_mlspike GRAPH SUMMARY',...
        'handlevisibility','off')
    ha = [];
else
    ha = flipud(findall(hf,'type','axes'))';
    if length(ha)~=4
        clf(hf)
        ha = [];
    end
end
if isempty(ha)
    ha = zeros(1,4);
    for i=1:4
        ha(i) = subplot(4,1,i,'parent',hf); 
        colormap(ha(i),'gray') % cannot do colormap(hf,'gray') on old Matlab versions
    end
    set(ha,'xtick',[],'ytick',[],'box','on')
end
    
%---
function s = logsumexp(X, dim)
% function s = logsumexp(X, dim)
%---
% Compute -log(sum(exp(-X),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com), modified by Thomas Deneux
% 
% See also logmultexp

if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each dim
xmax = min(X,[],dim);
X = fn_subtract(X,xmax);
s = xmax-log(sum(exp(-X),dim));
s(isinf(xmax)) = Inf;

%---
function Y = logmultexp(W,X)
% function Y = logmultexp(W,X)
%---
% Compute -log(W*exp(-X)) while avoiding numerical underflow.
% 
% See also logsumexp

% X = m + X1
m = min(X);
X1 = fn_subtract(X,m);

% exp(X) = exp(m) * exp(X1)
X1e = exp(-X1);

% W*exp(X) = exp(m) * W*exp(X1)
Y1e = W*X1e;

% log(W*exp(X)) = m + log(W*exp(X1))
Y = fn_subtract(m,log(Y1e));
Y(:,isinf(m)) = Inf;

%---
function Y = logmultexp_column(W,X)
% function Y = logmultexp_column(W,X)
%---
% Compute -log(exp(-X)*W) while avoiding numerical underflow.
% 
% See also logsumexp

% X = m + X1
m = min(X,[],2);
X1 = fn_subtract(X,m);

% exp(X) = exp(m) * exp(X1)
X1e = exp(-X1);

% exp(X) = exp(m) * exp(X1)*W
Y1e = X1e*W;

% log(exp(X)*W) = m + log(exp(X1)*W)
Y = fn_subtract(m,log(Y1e));
Y(isinf(m),:) = Inf;

%---
function varargout = logsample(l,nsample_flag)
% function [ii jj ...] = logsample(l,nsample|flag)
%---
% draw samples from the negative log probability distribution

if nargin<2, nsample_flag=1; end

if isnumeric(nsample_flag)
    p = log2proba(l);
    nsample = nsample_flag;
    idx = 1 + sum(bsxfun(@gt,rand(1,nsample),cumsum(p(:))));
    switch nargout
        case 1
            if ~isvector(l), error 'two outputs when l is a matrix', end
            varargout = {idx};
        case 2
            [ii jj] = ind2sub(size(l),idx);
            varargout = {ii jj};
    end
elseif strcmp(nsample_flag,'rows')
    % each row is a different distribution
    p = log2proba(l,2);
    nsample = size(l,1);
    idx = 1 + sum(bsxfun(@gt,rand(nsample,1),cumsum(p,2)),2);
    varargout = {idx};
end
