function varargout = spk_calcium(varargin)
% function [F F0 drift] = spk_calcium(spikes,par|dt+other parameters) 
% function par = spk_calcium('par'[,'3exps'][,dt[,T]][,other parameters]) 
%---
% 
% Input:
% - spikes  vector, array or cell array - spike times or spike counts
% - dt      frame acquisition time
% - T       final time
% - '3exps' get parameter defaults for 3 exponentials calcium response as
%           in Grewe et al., 2010
% 
% Input/Output:
% - par     parameter structure
%
% Output:
% - F       column vector or array (one signal per column) - predicted
%           fluorescence signal, after noise has been added
% - F0      fluorescence signal, before noise has been added
% - drift   drift

if nargin==0, help spk_calcium, return, end
 
% Default parameter
if ischar(varargin{1})
    if ~strcmp(varargin{1},'par'), error argument, end
    par = defaultpar(varargin{2:end});
    varargout = {par};
    return
end

% Convert to cell
spikes = varargin{1};
if ~iscell(spikes)
    if isvector(spikes) || isempty(spikes)
        spikes = {spikes};
    else
        spikes = num2cell(spikes,1);
    end
end
ndata = numel(spikes);
nt = [];
if ~isempty(spikes{1}) && ~any(mod(spikes{1},1)) && length(spikes{1})>=5 && any(spikes{1}==0)
    % input is a spike count -> we know how many discrete times there are
    nt = length(spikes{1}); 
end

% Parameters
if nargin==2 && isstruct(varargin{2})
    par = fn_structmerge(defaultpar,varargin{2},'skip');
else
    par = defaultpar(varargin{2:end});
end

% Convert to spike times
spikes = fn_timevector(spikes,par.dt,'times');
if ~iscell(spikes), spikes = {spikes}; end

% T max
if isempty(par.T)
    if isempty(nt)
        par.T = max([spikes{:}])+1;
    else
        par.T = nt*par.dt;
    end
else
    if ~isempty(nt) && par.T~=nt*par.dt
        error 'par.T is not consistent with length of spikes input(s)'
    end
end
 
% Go
if ndata==1
    [F F0 drift] = forward(spikes{1},par);
else
    [F F0 drift] = deal(cell(1,ndata));
    dt = par.dt; T = par.T;
    for k=1:ndata
        if ~isscalar(dt), par.dt=dt(k); end
        if ~isscalar(T), par.T=T(k); end
        [F{k} F0{k} drift{k}] = forward(spikes{k},par);
    end
end
varargout = {F F0 drift};


%-------------------------------------------------------------------------%
%                       PARAMETERS                                        %
%-------------------------------------------------------------------------%

function par = defaultpar(varargin)

% User input
p = struct;
i = 0;
while i<length(varargin)
    i = i+1;
    a = varargin{i};
    if isnumeric(a)
        if ~isfield(p,'dt')
            p.dt = a;
        else
            p.T = a;
        end
    else
        switch a
            case {'1exp' '3exps'}
                p.type = a;
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

% Mandatory parameter
par.dt = [];

% Other time parameter
par.T = [];

% Physiological parameters
if isfield(p,'type')
    par.type = p.type;
else
    par.type = '1exp';
end
par.F0 = 1;
switch par.type
    case '1exp'
        par.delay = 0;
        par.a = .1;
        par.tau = 1;
    case '3exps'
        par.delay = 0.001;
        par.a = [.077 .031];
        par.tau = [0.0081 0.056 0.777];
end
par.ton = 0;
par.saturation = 0;
par.pnonlin = []; % non-linear polynomial output
par.sigma = 0;
par.hill = 1;
par.c0 = 0; % calcium baseline level in "individual spike transient amplitude" unit; needs to be set only when par.hill~=1

% Initial state
par.x0 = 0;

% Drifts
% par.drift.method = 'state';
par.drift.method = 'basis functions';
par.drift.effect = 'multiplicative';
par.drift.parameter = 0;

% user input
par = fn_structmerge(par,p,'strict','recursive');

%-------------------------------------------------------------------------%
%                       FORWARD                                           %
%-------------------------------------------------------------------------%

function [F F0 drift] = forward(spikes,par)

dt = par.dt;
a = par.a;
tau = par.tau;
ton = par.ton;
hill = par.hill; c0 = par.c0;
sat = par.saturation;
pnonlin = par.pnonlin;
nt = round(par.T/dt);

% add a delay for spike to act
spikesactive = spikes + par.delay;

% go
switch par.type
    case '1exp'
        % precompute by how much x will increase at each time step because of
        % calcium influx
        increase = zeros(1,nt);
        for k=1:length(spikesactive)
            tk = spikesactive(k);
            ik = 1+ceil(tk/dt);
            if ik>=1 && ik<=nt
                increase(ik) = increase(ik) + exp(-((ik-1)*dt-tk)/tau);
            end
        end
        % dynamic system
        decay = exp(-dt/tau);
        ct = fn_switch(isempty(par.x0),0,par.x0);
        c = zeros(nt,1);
        for i=1:nt
            ct = ct*decay + increase(i);
            c(i) = ct;
        end
        % probe
        cn = (c0+c).^hill-c0^hill;
        p = cn ./ (1+sat*cn);
        if ton>0
            ptarget = p;
            pspeed = (1+sat*cn)/ton;
            p = zeros(nt,1);
            pt = 0;
            for t=1:nt
                pt = ptarget(t) + (pt-ptarget(t))*exp(-pspeed(t)*dt);
                p(t) = pt;
            end
        end
        % ad-hoc nonlinearity
        if ~isempty(pnonlin)
            p = polyval([fliplr(pnonlin) 1-sum(pnonlin) 0],p);
        end
        % measure
        ypred = a*p;
    case '3exps'
        % convolution with a 3-exponentials canonic function
        if par.ton>0
            error 'not implemented yet, and not clear which model to use'
        end
        tt = (0:nt-1)'*dt;
        a1 = a(1);
        a2 = a(2);
        ton = tau(1);
        t1 = tau(2);
        t2 = tau(3);
        ypred = zeros(nt,1);
        for i=1:length(spikesactive)
            tti = tt-spikesactive(i);
            idx = (tti>0 & tti<5);
            tti = tti(idx);
            ypred(idx) = ypred(idx) + (tti>0).*(1-exp(-tti/ton)).*(a1*exp(-tti/t1)+a2*exp(-tti/t2));
        end
        if ~isempty(par.pnonlin), error 'nonlinear output not possible with 3 exponentials', end
end


% add a baseline of 1 and scale by baseline fluorescence
F0 = (1+ypred)*par.F0;

% add drifts
if par.drift.parameter
    switch par.drift.method
        case 'state'
            innovationnoise = par.drift.parameter(1) * randn(nt,1);
            memory = exp(-(0:dt:40)/10)'; % return to baseline after 10s
            drift = conv(innovationnoise,memory);
            drift = drift(1:nt);
        case 'basis functions'
            ndrift = par.drift.parameter(1);
            drifts = linspace(-1,1,nt)'; % RMS = 1/sqrt(3)
            if ndrift>1
                phase = linspace(0,2*pi,nt)';
                nsin = (ndrift-1)/2;
                drifts(1,1+2*nsin) = 0; % pre-allocate
                for k=1:nsin
                    drifts(:,2*k) = sin(k*phase); % RMS = 1/sqrt(2)
                    drifts(:,2*k+1) = cos(k*phase); % RMS = 1/sqrt(2)
                end
            end
            if isscalar(par.drift.parameter), amp = .1; else amp = par.drift.parameter(2); end
            beta = amp * randn(ndrift,1);
            drift = drifts * beta;
        otherwise
            error 'unknown drift method'
    end
else
    drift = zeros(nt,1);
end
switch par.drift.effect
    case 'additive'
        ypred = ypred + drift;
    case 'multiplicative'
        ypred = (1+ypred) .* (1+drift) - 1;
    otherwise
        error 'unknown drift method'
end

% add noise
if par.sigma
    ypred = ypred + randn(nt,1)*par.sigma; 
end

% add a baseline of 1 and scale by baseline fluorescence
F = (1+ypred)*par.F0;


