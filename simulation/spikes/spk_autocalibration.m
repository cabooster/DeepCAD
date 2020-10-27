function varargout = spk_autocalibration(varargin)
% function [tau amp sigma events] = spk_autocalibration(calcium,dt|pax)
% function [sigma] = spk_autocalibration(calcium,dt|pax,'sigmaonly')
% function pax = spk_autocalibration('par',dt,other parameters...)
%---
% 
% Input:
% - calcium     calcium signals
% - 'sigmaonly' flag for estimating only sigma; note that it is more
%               appropriate to call function spk_autosigma directly
% 
% Input/Output:
% - pax         structure with autocalibration parameters
%
% Output:
% - tau,amp,sigma   estimated values for transient decay time and amplitude
%               and noise std in the data
% - events      a description of the events detected that yielded the
%               estimation of 'amp'; the time and number of assigned spikes
%               are given


% Default parameter
if ischar(varargin{1})
    if ~strcmp(varargin{1},'par'), error argument, end
    pax = defaultpar(varargin{2:end});
    varargout = {pax};
    return
end

% Auto-calibration
calcium = varargin{1};
if ~iscell(calcium), calcium = {calcium}; end
if isstruct(varargin{2})
    p = varargin{2};
    pax = defaultpar('intern');
    if isfield(p,'mlspikepar'), pax.mlspikepar = p.mlspikepar; end % prevent error in next line
    pax = fn_structmerge(pax,p,'strict','recursive');
    if nargin>=3
        if ~strcmp(varargin{3},'sigmaonly'), error argument, end
        sigmaonly = true;
    else
        sigmaonly = false;
    end
else
    sigmaonly = strcmp(varargin{end},'sigmaonly');
    if sigmaonly, varargin(end)=[]; end
    pax = defaultpar(varargin{2:end});
end
[tau amp sigmaest events] = autocalibration(calcium,pax,sigmaonly); 
if sigmaonly
    varargout = {sigmaest};
else
    varargout = {tau amp sigmaest events};
end

function [tau amp sigmaest eventdesc] = autocalibration(calcium,pax,sigmaonly)

% input
if ~iscell(calcium), calcium = {calcium}; end
calcium = fn_map(@column,calcium);
ntrial = length(calcium);
for k=1:ntrial, calcium{k} = double(calcium{k}); end % needed for call to fmincon later
dt = pax.dt;
if isscalar(dt), dt = repmat(dt,1,ntrial); end
[realspikes reala realtau] = deal(pax.realspikes,pax.reala,pax.realtau);
if isempty(realspikes), realspikes = cell(1,ntrial); end
if ~iscell(realspikes), realspikes = {realspikes}; end

% some old parameters are now assigned fixed values
pax.est_tau_fulldata = false;
pax.handlenonlinear = true;
pax.domove12border = false;

% prepare display
dodisplay = fn_ismemberstr(pax.display,{'steps' 'pause' 'save'}) && ~sigmaonly;
dohisto = strcmp(pax.display,'histo');
dopause = strcmp(pax.display,'pause');
dosave = strcmp(pax.display,'save');
if dodisplay
    if dosave, hf = pax.figsave{1}; else hf = 691; end
    if ~ishandle(hf), figure(hf), end
    clf(hf)
    set(hf,'numbertitle','off','name','Auto-calibration','handlevisibility','off')
    if dosave
        hp1 = hf; hp2 = hf;
    else
        hp1 = uipanel('parent',hf,'pos',[0 .5 1 .5]);
        hp2 = uipanel('parent',hf,'pos',[0 0  1 .5]);
    end
    drawnow
elseif dohisto
    hahisto = pax.hahisto;
    if isempty(hahisto)
        hf = 691; if ~ishandle(hf), figure(hf), end
        clf(hf), set(hf,'numbertitle','off','name','Auto-calibration','handlevisibility','off')
        hahisto = axes('parent',hf);
    elseif fn_isfigurehandle(hahisto)
        clf(hahisto)
        hahisto = axes('parent',hahisto);
    end
end
if dosave
    figsavename = pax.figsave{2};
end

%% estimate noise level

% s = pax.ynoise;
% if isstruct(s) && s.ynoise1~=0 && dodisplay, disp 'estimate noise level', end
% 
% sigmaest = zeros(1,ntrial);
% for k=1:ntrial
%     x = calcium{k};
%     if isnumeric(s)
%         noiseestk = s;
%     else
%         noiseestk = s.ynoise0;
%         if s.ynoise1
%             noiseestk = noiseestk + s.ynoise1*std(fn_filt(x/mean(x),[s.f1 s.f2]/dt(k),'b'));
%         end
%         noiseestk = fn_coerce(noiseestk,s.ynoisemin,s.ynoisemax);
%     end
%     sigmaest(k) = noiseestk / sqrt(dt(k));
% end
% sigmaest = mean(sigmaest);
% %fprintf('ynoise: %f\n',sigmaest*mean(sqrt(dt)))

sigmaest = spk_autosigma(calcium,dt,pax.autosigmasettings);
if dodisplay, fprintf('estimated sigma: %.2g\n',sigmaest), end

% return if estimation of sigma only
if sigmaonly
    [tau amp eventdesc] = deal([]);
    return
end

sigmaest_events = sigmaest;


%% detect 'events'
if dodisplay && pax.saturation && fn_dodebug, disp 'detect events: no nonlinearity here!', end

par = tps_mlspikes('par',dt);
par.a = pax.eventa; % this value of a is not arbitrary but will influence the level of nonlinearity
par.tau = pax.eventtau;

% (no nonlinearity here!)
% par.saturation = pax.saturation;
% par.pnonlin = pax.pnonlin;
% par.hill = pax.hill;
% par.c0 = pax.c0;

%par.algo = fn_structmerge(par.algo,'cmax',8,'nc',40,'nb_driftslope',25,'ns',25);
par.special.nonintegerspike_minamp = pax.nonintegerspike_minamp;
par.drift.parameter = pax.driftparam;
par.drift.baselinestart = pax.baselinestart;
par.finetune.spikerate = pax.eventrate;
par.finetune.sigma = sigmaest_events;
par.display = 'none';
par = fn_structmerge(par,pax.mlspikepar,'recursive');

[n nn fit drift] = deal(cell(1,ntrial));

% save time: do not repeat previous computation
if pax.dosaveprecomp
    H = fn_hash({calcium,par,dt},8);
    fsave = fn_cd('spikes','save','autocalibration_precomp',['spk_autocalibration_firststep_' H '.mat']);
    okf = exist(fsave,'file');
    if okf
        if dodisplay, disp 'load first step from file', end
        [n,fit,drift] = fn_loadvar(fsave);
    end
end

for i=1:ntrial
    if ~pax.dosaveprecomp || ~okf
        par.dt = dt(i);
        [n{i} fit{i} dum dum dum drift{i}] = tps_mlspikes(calcium{i},par); %#ok<ASGLU>
    end
    nn{i} = round(n{i}*par.a*100);
    if dodisplay && (~pax.dosaveprecomp || ~okf || (i==ntrial))
        idx = max(1,i-19):i;
        spk_display(dt(idx),{realspikes(idx) nn(idx)},{calcium(idx) fit(idx) drift(idx)}, ...
            'in',hp1,'stats','none')
        drawnow
    end
end
if pax.dosaveprecomp && ~okf
    if dodisplay, disp 'save first step to temporary file to save time for next estimation with the same data', end
    fn_savevar(fsave,n,fit,drift)
end

if dodisplay && dopause
    spk_display(dt,{realspikes nn},{calcium fit drift},'stats','none','in',hp1)
    pause
end

%% cut isolated events of reasonable amplitude
if dodisplay, disp 'sub-select events', end

maxamp = pax.maxamp; % more parameters!!!!
tspan = pax.eventtspan;
tbef = pax.tbef;
cmax = pax.cmax;
taft = pax.taft;

% windows = cell(1,0);
anyevent = false(1,ntrial);
[events keptevents keptnn modcalcium] = deal(cell(1,ntrial));
for i=1:ntrial
    % sort out isolated events
    T = dt(i)*length(calcium{i});
    kevent = find(logical(n{i})); % time indices of events
    eventsi = kevent*dt(i);        % time of events
    events{i} = eventsi;
    lastevent = 0;
    nevent = length(eventsi);
    okevent = false(1,nevent);
    keptnn{i} = zeros(length(nn{i}),1);
    j=0;
    while j<nevent
        % j will be the last event of group of events jj
        % idxbef, idx, idx1 and idxmax are, respectively, the time indices
        % just before the first event, of all events, of center of mass and
        % of last event of the group
        % tj is the time of the first event 
        j = j+1;
        tj = eventsi(j);
        if tj-lastevent<=tbef, lastevent=tj; continue, end % event is too close to previous one
        lastevent=tj;
        idxbef = kevent(j)-1;
        if (j<nevent && eventsi(j+1)-tj<=tspan)
            % merge events that are close to each other by less than tspan
            nmerge = find(eventsi(j+1:end)-tj<=tspan,1,'last');
            jj = j+(0:nmerge); % time indices of events to be merged
            idx = kevent(jj);
            ni = sum(n{i}(idx)); % amplitude of current event
            idx1 = round(sum(idx.*n{i}(idx))/ni); % time position (index) of merged event obtained as a weighted sum
            idxmax = kevent(j+nmerge); % time index at which maximum calcium is reached
            j = j+nmerge; % jump to the last of the merged events
            lastevent = eventsi(j);
        else
            jj = j;
            idx1 = kevent(j);
            ni = n{i}(idx1);
            idxmax = kevent(j);
        end
        if T-tj<=taft || (j<nevent && eventsi(j+1)-tj<=taft)
            if j<nevent, j=find(eventsi-tj<=taft,1,'last'); end % skip all events that are too close
            continue
        end % next event is too close
        if fit{i}(idxbef)-drift{i}(idxbef)>cmax, 
            continue
        end % calcium level is too high when event starts
        if n{i}(idxmax)>maxamp
            continue
        end % event is too large, strong nonlinearities could occur
        okevent(jj) = true;
        keptevents{i}(end+1) = idx1*dt(i);
        keptnn{i}(idx1) = round(ni*par.a*100);
    end
    
    % modified calcium signals with only the kept events
    anyevent(i) = any(okevent);
    if ~anyevent(i), continue, end
    othern = n{i}; othern(kevent(okevent))=0;
    p = par; p.F0 = 1; p.dt = dt(i);
    othercalcium = tps_mlspikes(othern,p); % only tps_mlspike accepts non-integer spikes, contrary to spk_calcium 
    modcalcium{i} = calcium{i}-drift{i}.*(othercalcium-1);
end

% no event found?
idx = find(anyevent);
if isempty(idx)
    disp 'auto-calibration failed: no isolated event found!'
    [tau amp eventdesc] = deal([]);
    return
end

% display
if dodisplay
    spk_display(dt,{realspikes keptnn},{calcium fit drift},'stats','none','in',hp1)
    if dopause, pause, end
    if dosave, print(hf,'-dpsc2',[figsavename '.ps'],'-append'), end
    spk_display(dt(idx),{realspikes(idx) keptnn(idx)},{modcalcium(idx)},'stats','none','in',hp2)
    drawnow
    if dopause, pause, end
end

% restrict the analysis to kept trials
idxanykeptevent = find(fn_map(@length,keptevents));
keptevents1 = keptevents(idxanykeptevent);
realspikes1 = realspikes(idxanykeptevent);
modcalcium1 = modcalcium(idxanykeptevent);
dt1 = dt(idxanykeptevent);
ntrial1 = length(idxanykeptevent);

%% estimate tau and amplitude of individual events

% high-pass filter the (modified) calcium signals
tdrift = pax.tdrift;
[modcalciumf1 modcalciumflow1] = deal(cell(1,ntrial1));
for i=1:ntrial1
    modcalciumf1{i} = fn_filt(modcalcium1{i},tdrift/dt1(i),'hmd');
    modcalciumflow1{i} = modcalcium1{i}-modcalciumf1{i};
end

% estimate tau on original or modified calcium signals?
if pax.est_tau_fulldata
    if dodisplay, disp 'estimate tau (on full data) - no nonlinearity here!', end
    % (high-pass filter the original calcium signals)
    calciumf = cell(1,ntrial);
    for i=1:ntrial
        calciumf{i} = fn_filt(calcium{i},tdrift/dt(i),'hmd');
    end
    % (estimation)
    opt = optimset('Algorithm','active-set', ...
        'maxfunevals',10000,'tolx',1e-5,'tolfun',1e-8, ...
        'display','none'); %,'PlotFcns',{@optimplotx,@optimplotfval,@optimplotstepsize,@optimplotconstrviolation});
    pstart = pax.eventtau;
    LB = pax.taumin;
    UB = pax.taumax;
    FACT = 1e-2;
    % (no nonlinearity here!)
    pfwd0 = spk_calcium('par','a',pax.eventa ...
        ); %,'pnonlin',pax.pnonlin,'saturation',pax.saturation,'hill',pax.hill,'c0',pax.c0);
    tau = fmincon(@(p)energy(p/FACT,events,calciumf,dt,tdrift,pfwd0),pstart*FACT,[],[],[],[],LB*FACT,UB*FACT,[],opt)/FACT;
else
    if dodisplay, disp 'estimate tau - no nonlinearity here!', end
    opt = optimset('Algorithm','active-set', ...
        'maxfunevals',10000,'tolx',1e-5,'tolfun',1e-8, ...
        'display','none'); %,'PlotFcns',{@optimplotx,@optimplotfval,@optimplotstepsize,@optimplotconstrviolation});
    pstart = pax.eventtau; if fn_dodebug, disp 'line was modified since gcamp6s estimations!', end
    LB = pax.taumin;
    UB = pax.taumax;
    FACT = 1e-2;
    % (no nonlinearity here!)
    pfwd0 = spk_calcium('par','a',pax.eventa ...
        ); %,'pnonlin',pax.pnonlin,'saturation',pax.saturation,'hill',pax.hill,'c0',pax.c0);
    tau = fmincon(@(p)energy(p/FACT,keptevents1,modcalciumf1,dt1,tdrift,pfwd0),pstart*FACT,[],[],[],[],LB*FACT,UB*FACT,[],opt)/FACT;
end

% get the amplitude of individual events
[dum fit1 amps] = energy(tau,keptevents1,modcalciumf1,dt1,tdrift,pfwd0); %#ok<ASGLU>

% another sub-selection of events!
nn1 = cell(1,ntrial1);
for i=1:ntrial1
    amps{i} = amps{i}*pfwd0.a; % real amplitude rather than relative to pfwd.a
    okevent = (amps{i}>=applynonlinearity(1,pax)*pax.amin); % keep only event whose newly estimated amplitude is above amin
    keptevents1{i} = keptevents1{i}(okevent);
    amps{i} = amps{i}(okevent);
    nti = length(modcalcium1{i});
    nn1{i} = zeros(1,nti);
    nn1{i}(round(keptevents1{i}/dt1(i))) = round(amps{i}*100);
    fit1{i} = fit1{i}+modcalciumflow1{i};    
end

% check again that not all events where thrown out
if all(fn_isemptyc(keptevents1))
    disp 'auto-calibration failed: no isolated event found!'
    [tau amp] = deal([]);
    return
end

if dodisplay
    spk_display(dt1,{realspikes1 nn1},{modcalcium1 fit1},'stats','none','in',hp2)
    drawnow
    if dopause, pause, end
end


%% assign number of spikes for each event

if dodisplay, disp 'histogram and assign number of spikes to events', end

% histogram of event amplitudes 
da = .001;
aa = (0:da:applynonlinearity(6,pax)*pax.amax); % event amplitudes, not values of a
na = length(aa);
allamps = fn_timevector(cat(1,amps{:}),aa); % 'histogram' of event amplitudes

% filter and other processing
m = (aa(:)>=pax.amin);
if ~isempty(pax.histosmooth)
    sigmaa = pax.histosmooth;
else
    % was 0.05 in the original code for OGB, but this is not smoothing
    % enough in the case of gcamp6s; here is a small heuristic to
    % automatically set its value
    sigmaa = applynonlinearity(1,pax)*max(pax.amax/2,geo_mean([pax.amin pax.amax])); 
end
allampsf = fn_filt(allamps,sigmaa/da,'mask',m); 
allampsff = allampsf ./ (fn_filt(allamps,2*sigmaa/da,'mask',m)+1e-6); % +1e-6 to avoid NaNs
idx = find(aa>=applynonlinearity(1,pax)*pax.amin & aa<=applynonlinearity(1,pax)*pax.amax);
allamps3 = zeros(na,1);
spikescaling = applynonlinearity(1:3,pax)/applynonlinearity(1,pax);
allamps3(idx) = pax.costfactor*interp1(aa,allampsff,column(spikescaling(1:3))*row(aa(idx)));

% determine single-spike amplitude
[dum midx] = max(allamps3);  %#ok<ASGLU>
a1spike = aa(midx); 
amp0 = a1spike/applynonlinearity(1,pax); % estimated parameter a at this stage

% assign number of spikes for each event (need to go back and forth with
% the nonlinearity)
asep = amp0 * applynonlinearity(1.3:20.3,pax); % change 18/06/2015
if pax.domove12border
    % (try to find a better separation between 1 and 2 spikes)
    test = [0; diff(allampsf,2)]; test(aa<=a1spike | aa>=2*a1spike)=0;
    [dum sepidx] = max(test); %#ok<ASGLU>
    asep(1) = aa(sepidx);
end
asep(1) = min(asep(1),pax.amax*applynonlinearity(1.15,pax)); % change 18/06/2015
eventdesc = struct('time',cell(1,ntrial1),'amp',cell(1,ntrial1),'number',cell(1,ntrial1));
if ntrial1~=length(keptevents1), error programming, end
spikes1 = cell(1,ntrial1);
for i=1:ntrial1
    spikes1{i} = [];
    nspki = length(keptevents1{i});
    if length(amps{i})~=nspki, error programming, end
    eventdesc(i).time = keptevents1{i};
    eventdesc(i).amp = amps{i};
    eventdesc(i).num = zeros(1,nspki);
    for j=1:nspki
        nspj = find(amps{i}(j)<=asep,1,'first');
        if isempty(nspj)
            error 'a detected calcium event was assigned a burst of more than 20 spikes, this might degrade the autocalibration accuracy, please reduce parameter ''maxamp'' to avoid such events to be considered'
        end
        spikes1{i} = [spikes1{i} keptevents1{i}(j)*ones(1,nspj)];
        eventdesc(i).num(j) = nspj;
    end
end

    
%% estimate amplitude of single spikes

% % forward prediction
% %[useless?] ndata = length(spikes1);
% Fpred = cell(1,ndata);
% for i=1:ndata
%     pfwd = spk_calcium('par','dt',dt1(i),'T',length(calciumf{i})*dt1(i), ...
%         'tau',tau,'a',1,'pnonlin',pax.pnonlin, ...
%             'saturation',pax.saturation,'hill',pax.hill,'c0',pax.c0);
%     Fpred{i} = spk_calcium(spikes1{i},pfwd);
% end
% 
% % high-pass filter
% for i=1:ndata, Fpred{i} = fn_filt(Fpred{i},tdrift/dt(i),'hmd'); end
% 
% % fit
% amp = cat(1,Fpred{:})\cat(1,calciumf{:});
% amp = fn_coerce(amp,pax.amin,pax.amax);
% fit = cell(1,ndata); 
% for i=1:ndata, fit{i} = amp*Fpred{i}+(calcium1{i}-calciumf{i}); end

%% final fine re-estimation of a and tau


% forward parameters
pfwd = spk_calcium('par');
pfwd.dt = dt1;
nt = fn_map(@length,modcalciumf1);
pfwd.T = nt.*dt1;
pfwd.saturation = pax.saturation;
pfwd.pnonlin = pax.pnonlin;
pfwd.hill = pax.hill;
pfwd.c0 = pax.c0;

% estimation
opt = optimset('Algorithm','interior-point', ... note that previous choice of 'active-set' was sometimes getting stuck in "not even local minima"!!
    'maxfunevals',10000,'tolx',1e-20,'tolfun',1e-5, ...
    'display','none');
pstart = [amp0 tau];
LB = [pax.amin pax.taumin];
UB = [pax.amax pax.taumax];
FACT = [1e-2 1e-2];

if pax.est_tau_fulldata
    if dodisplay, disp 're-estimate A', end
    % keep the previous estimate of tau, estimate a only
    pvalues = fmincon(@(p)energycalib([p./FACT(1) tau],spikes1,modcalcium1,pfwd,tdrift),pstart(1).*FACT(1),[],[],[],[],LB(1).*FACT(1),UB(1).*FACT(1),[],opt)./FACT(1);
    amp = pvalues;
else
    if dodisplay, disp 're-estimate A and tau', end
    pvalues = fmincon(@(p)energycalib(p./FACT,spikes1,modcalcium1,pfwd,tdrift),pstart.*FACT,[],[],[],[],LB.*FACT,UB.*FACT,[],opt)./FACT;
    amp = pvalues(1);
    tau = pvalues(2);
end
[e fit2 drift2] = energycalib([amp tau],spikes1,modcalcium1,pfwd,tdrift); %#ok<ASGLU>

% [e fit drift] = energycalib(pvalues,spikes1,calcium1,pfwd,pax.tdrift); %#ok<ASGLU>


%% display

if dodisplay
    idxrep = [1:ntrial1 1];
    spk_display(dt1(idxrep),{realspikes1(idxrep) spikes1(idxrep)}, ...
        {modcalcium1(idxrep) fit2(idxrep) drift2(idxrep)},'stats','none','in',hp2)
    ha = findobj(hp2,'type','axes');
    hahisto = ha(1);
end
if dodisplay || dohisto
    plot(aa,[allamps/max(allamps) allampsf/max(allampsf) allampsff/max(allampsff) allamps3/max(allamps3)],'parent',hahisto)
    axis(hahisto,'tight')
    xlabel(hahisto,'transient amplitude (DF/F)')
    midxm = midx:midx:na;
    spikescaling = applynonlinearity(1:3,pax);
    try
        line(amp0*spikescaling,allampsff(1+round(amp0*spikescaling/da))/max(allampsff),'linestyle','none', ...
            'marker','.','markersize',14,'color','c','parent',hahisto)
        line(amp*spikescaling,allampsff(1+round(amp*spikescaling/da))/max(allampsff),'linestyle','none', ...
            'marker','.','markersize',14,'color','k','parent',hahisto)
        line(reala*spikescaling,allampsff(1+round(reala*spikescaling/da))/max(allampsff),'linestyle','none', ...
            'marker','.','markersize',14,'color','r','parent',hahisto)
    end
    for i=1:length(asep), line(asep(i)*[1 1],[0 1],'color','k','parent',hahisto), end
    xlim(hahisto,[0 applynonlinearity(3,pax)*pax.amax])
    str = sprintf('estimated a: %.1f%%(real: %.1f%%), tau: %.2fs(real: %.2fs)',amp*100,reala*100,tau,realtau);
    if dodisplay
        ha = axes('pos',[.4 .96 .2 .02],'visible','off','parent',hf);
        ht = text(.5,.5,str,'parent',ha, ...
            'horizontalalignment','center','fontweight','bold','fontsize',14); %#ok<NASGU>
    else
        title(hahisto,str)
    end
    drawnow
    if dopause, pause, end
    if dosave, print(hf,'-dpsc2',[figsavename '.ps'],'-append'), end
end

%---
function [e Fpred amps] = energy(tau,events,F,dt,tdrift,pfwd0)

% parameters and sizes
pfwd = pfwd0;
pfwd.tau = tau;
ndata = numel(events);
nt = fn_map(@length,F,'array');
if isscalar(dt), dt = repmat(dt,[1 ndata]); end

% forward prediction
[dif Fpred amps] = deal(cell(1,ndata));
for i=1:ndata
    % build regressors
    nevent = length(events{i});
    A = zeros(nt(i),nevent);
    pfwd.dt = dt(i);
    pfwd.T = nt(i)*dt(i);
    for j=1:nevent        
        A(:,j) = spk_calcium(events{i}(j),pfwd);
    end
    % high-pass filter
    A = fn_filt(A,tdrift/dt(i),'hmd');
    % fit
    amps{i} = (A\F{i});
    Fpred{i} = A*amps{i};
    dif{i} = F{i}-Fpred{i};
end

% error
e = sqrt(mean(cat(1,dif{:}).^2))*100;

%---
function [e fit drift] = energycalib(p,spikes,F,pfwd0,tdrift)

% parameters and sizes
ndata = numel(spikes);
nt = fn_map(@length,F,'array');
pfwd0.a = p(1);
pfwd0.tau = p(2);
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
    if isequal(tdrift,0)
        drift{i} = mean(base)*ones(nt,1);
    elseif strcmp(tdrift,'trend')
        drift{i} = base - detrend(base);
    else
        drift{i} = fn_filt(base,tdrift/pfwd0.dt(i),'lmd',1); 
    end
    fit{i} = drift{i}.*Fpred0{i};
    dif{i} = F{i}-fit{i};
end

% error
e = sqrt(mean(cat(1,dif{:}).^2))*100;

% % display
% fn_figure('energycalib')
% spk_display(pfwd0.dt,spikes,{F fit drift})

%---
function values = applynonlinearity(values,pax)

% handling nonlinearity: the amplitude for n spikes is not exactly n times
% the amplitude for 1 spike
if ~pax.handlenonlinear, return, end

if ~isempty(pax.pnonlin)
    p = [fliplr(pax.pnonlin) 1-sum(fliplr(pax.pnonlin)) 0];
    values = polyval(p,values);
else
    cn = (pax.c0+values).^pax.hill - pax.c0^pax.hill;
    values = cn ./ (1 + pax.saturation*cn);
end

%-------------------------------------------------------------------------%
%                       PARAMETERS                                        %
%-------------------------------------------------------------------------%

function pax = defaultpar(varargin)

% dt
pax.PART1 = 'PLEASE SET PARAMETER DT';
pax.dt = [];

% range for amplitude and tau
pax.PART2 = 'RANGE FOR A AND TAU';
pax.amin = .04;
pax.amax = .1;
pax.taumin = .4;
pax.taumax = 1.6;

% sigma estimation
pax.PART3 = 'SIGMA ESTIMATION (see spk_autosigma.m)';
pax.autosigmasettings = 'correlated';

% detect events
pax.PART4 = 'EVENTS DETECTION';
pax.eventa = .1;
pax.eventtau = .8;
pax.saturation = 0;
pax.pnonlin = [];
pax.hill = 1;
pax.c0 = 1;
pax.tdrift = 5;
pax.driftparam = .005;
pax.baselinestart = false;
pax.eventrate = .00001;
pax.nonintegerspike_minamp = .4;
pax.mlspikepar = struct;

% sub-select events
pax.PART5 = 'EVENTS SUB-SELECTION';
pax.maxamp = 2.5; % maximum height of detected event relative to pax.event a (i.e. when pax.eventa=.1, only events of amplitude<=.25 are kept)
pax.eventtspan = 0;
pax.tbef = 1;
pax.cmax = .01;
pax.taft = 1;

% histogram and assign number of spikes to events
pax.PART6 = 'HISTOGRAM AND ASSIGN SPIKES';
pax.histosmooth = []; % a heuristic will be used to set it if value remains empty
pax.costfactor = [1 .5 0]; %[1 2/3 1/3];

% display
pax.PART7 = 'DISPLAY PARAMETERS ';
pax.display = 'steps'; % 'none', 'steps', 'histo', 'pause' or 'save'
pax.figsave = {}; % figure number and file name (for 'save' mode)
pax.hahisto = []; % an axes handle where to display the histogram summary (for 'histo' mode)
% (real values are used for display only)
pax.realspikes = [];
pax.reala = 0;
pax.realtau = 0;

% User input
i = 0;
while i<length(varargin)
    i = i+1;
    a = varargin{i};
    if isnumeric(a)
        pax.dt = a;
    elseif strcmp(a,'intern')
        % add some 'internal' parameters that are used for debugging or
        % special actions but do not appear in the standard defaults
        pax.dosaveprecomp = false;
    else
        i = i+1;
        if ~isfield(pax,a), error argument, end
        pax.(a) = varargin{i};
    end
end

