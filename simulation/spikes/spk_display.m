function [stats hl] = spk_display(dt,spikes,calcium,varargin)
% function [stats hl] = spk_display(dt,spikes,calcium[,ylim][,'gridsize',nrowcol] ...
%               [,'in',axes handle][,'stats',statflag][,'cost',cost], ...
%               [,line options...][,'paper'][,other options...])
%---
% display calcium time courses with spikes
% 
% Input:
% - dt      scalar - time resolution
% - spikes  vector - a spike train
%           or cell array of vectors - several spike trains to compare
%           or cell array of cell arrays - multiple datasets, use one graph per dataset 
% - calcium vector - a calcium time courses
%           or array or cell array of vectors - several calcium time courses 
%           or cell array thereof - multiple datasets, use one graph per dataset 
% - ylim    clipping for calcium display
% - statflag    'none', 'global' [default], or 'local'
% - cost    parameter structure for spk_distance
% - line options        complex options can be specified for line display,
%           default line options are:
%           'color', {'rk' 'bkk'}       (1st element for spike, 2d for calcium) 
%           'linewidth', [.5 1 1]       (calcium only)
%           'fullline', [1 1 0]         (calcium only, 0 means dashed line)
% - 'paper' this flag leads to use different values for line widths and
%           marker sizes
% - other options
%           'displaymode',mode  different preset for font sizes, etc.
%           'burstdelay',value  specify time length for grouping spikes
%                               into a single number
%           'calciumevents' of 'calciumeventsfull'
%           'title',titl
%           'toptitle',titl
%           'rate'              second set of 'spikes' is in fact a spike
%                               rate rather than spike times
% 
% Output:
% - stats   statistics
% - hl      handles to (calcium) lines

if nargin==0, help spk_display, end

% Input
ylim = []; gridsize = []; in = []; color = [];
toptitl = []; titl = []; cost = []; burstdelay = []; calciumevents = [];
statflag = 'global'; lineoptions = struct; displaymode = 'screen';
spikemode = 'spikes';
k=0;
while k<length(varargin)
    k = k+1;
    a = varargin{k};
    if isnumeric(a)
        ylim = a;
    else
        switch a
            case 'in'
                k = k+1;
                in = varargin{k};
            case 'gridsize'
                k = k+1;
                gridsize = varargin{k};
            case {'color' 'linewidth' 'fullline'}
                k = k+1;
                lineoptions.(a) = varargin{k};
            case {'stat' 'stats'}
                k = k+1;
                statflag = varargin{k};
            case 'cost'
                k = k+1;
                cost = varargin{k};
            case 'title'
                k = k+1;
                titl = varargin{k};
            case 'toptitle'
                k = k+1;
                toptitl = varargin{k};
            case 'burstdelay'
                k = k+1;
                burstdelay = varargin{k};
            case {'calciumevents' 'calciumeventsfull'}
                k = k+1;
                calciumevents = varargin{k};
                calciumeventsmode = fn_switch(a,'calciumevents','simple','calciumeventsfull','full');
            case 'displaymode'
                k = k+1;
                displaymode = varargin{k};
            case {'paper' 'factorbox' 'smallsym'}
                displaymode = a;
            case 'rate'
                spikemode = 'rate';
            otherwise
                error argument
        end
    end
end
dorate = strcmp(spikemode,'rate');
  
% Read multiple formats for spikes and calcium
if isempty(dt), dt = 1; end % set arbitrary time constant (can happen when calcium is empty as well)
invertedorder = false;
% small format changes
if isempty(spikes)
    % no spike data -> get number of trials from the calcium!
    if isnumeric(calcium)
        if isvector(calcium)
            ntrial = 1;
        else
            ntrial = size(calcium,2);
        end
    elseif numel(calcium)>3 || isnumeric(calcium{1})
        ntrial = numel(calcium);
    else
        ntrial = numel(calcium{1});
    end
    spikes = repmat({{zeros(1,0)}},1,ntrial);
elseif isnumeric(spikes) && ~isempty(spikes) && ~isvector(spikes)
    % spikes from multiple trials in the form of an array of counts
    spikes = num2cell(spikes,1); 
end
% read spikes and guess how many data sets, how many spike trains per
% dataset
if isnumeric(spikes) && (isvector(spikes) || isempty(spikes))
    % 1 data set, 1 spike train
    if ~dorate, spikes = fn_timevector(spikes,dt,'times'); end % convert counts to spikes if necessary
    spikes = {{spikes}};
elseif isnumeric(spikes) && size(spikes,2)<=2
    % 1 data set, multiple spike trains
    if ~dorate, spikes = fn_timevector(spikes,dt,'times'); end % convert counts to spikes
    spikes = {spikes};
elseif isnumeric(spikes) && size(spikes,2)>2
    % multiple data set, 1 spike train per set
    if ~dorate, spikes = fn_timevector(spikes,dt,'times'); end % convert counts to spikes
    spikes = fn_map(@(x){x},spikes,'cell');
elseif ~iscell(spikes)
    error argument
elseif isnumeric(spikes{1}) && all(size(spikes{1})>1)
    error 'not implemented yet'    % spike counts
elseif isnumeric(spikes{1}) && numel(spikes)<=2 && isscalar(dt)
    % 1 data set, multiple spike trains
    if ~dorate, spikes = fn_timevector(spikes,dt,'times'); end
    spikes = {spikes};
elseif isnumeric(spikes{1}) && (numel(spikes)>2 || ~isscalar(dt))
    % multiple data sets, 1 spike train per set
    ngraph = numel(spikes);
    if isscalar(dt), dt = repmat(dt,[1 ngraph]); end
    if ~dorate, for i=1:ngraph, spikes{i} = fn_timevector(spikes{i},dt(i),'times'); end, end
    invertedorder = true;
    spikes = fn_map(@(x){x},spikes,'cell');
elseif iscell(spikes{1}) && numel(spikes)>2
    % multiple data sets, multiple trains per set, everything is ok
    ngraph = numel(spikes);
    if isscalar(dt), dt = repmat(dt,[1 ngraph]); end
    for i=1:ngraph, spikes{i} = fn_timevector(spikes{i},dt(i),'times'); end
elseif iscell(spikes{1}) && numel(spikes)<=2
    % multiple data sets, multiple trains per set, but cell arrays are
    % nested in the inverted order
    nspk = numel(spikes);
    if ~dorate, for i=1:nspk, spikes{i} = fn_timevector(spikes{i},dt,'times'); end, end
    invertedorder = true;
    spikes = cat(3,spikes{:});      % cell array of vectors
    spikes = num2cell(spikes,3);    % cell array of cell arrays, in the correct nesting order
    spikes = fn_map(@squeeze,spikes,'cell');
end
ngraph = numel(spikes);
if nargin<3 || isempty(calcium)
    calcium = cell(1,ngraph);
elseif ngraph==1
    % 1 data set
    if iscell(calcium)
        if iscell(calcium{1})
            for i=1:length(calcium), calcium(i) = calcium{i}(:); end
        end
        if ~isscalar(calcium)
            calcium = {fn_map(@column,calcium,'array')};
        end
    else
        if isvector(calcium)
            calcium = {calcium(:)};
        else
            calcium = {calcium};
        end
    end
elseif ~iscell(calcium)
    if isvector(calcium), error 'spikes data format indicated multiple datasets, but calcium data format indicated only one dataset', end
    calcium = num2cell(calcium,1);
elseif iscell(calcium{1})
    % multiple data sets, multiple calcium per data set given as cell arrays
    if numel(calcium)==ngraph && ~(numel(calcium{1})==ngraph && invertedorder)
        for i=1:ngraph, calcium{i} = [calcium{i}{:}]; end
    elseif numel(calcium{1})==ngraph
        calcium = fn_map(@column,calcium);
        calcium = cat(2,calcium{:});
        calcium = fn_map(@column,calcium);
        calcium = num2cell(calcium,2);
        for i=1:ngraph, calcium{i} = [calcium{i}{:}]; end
    else
        error 'spikes and calcium data sets do not match'
    end
else
    % multiple data sets, multiple calcium per data set (if any) given as arrays
    if numel(calcium)==ngraph && ~(size(calcium{1},2)==ngraph && invertedorder)
        if isvector(calcium{1})
            for i=1:ngraph, calcium{i} = calcium{i}(:); end
        end
    elseif size(calcium{1},2)==ngraph
        for i=1:numel(calcium), if iscell(calcium{i}), calcium{i} = [calcium{i}{:}]; end, end % convert cell arrays to numerical arrays if necessary
        calcium = cat(3,calcium{:});        % 3D numerical array
        calcium = num2cell(calcium,[1 3]);  % cell array of 2D arrays
        calcium = fn_map(@(x)squeeze(x),calcium);
    else
        error 'spikes and calcium data sets do not match'
    end
end

% time step
if isscalar(dt), dt = repmat(dt,[1 ngraph]); end
    
% Axes
if ngraph==1 && isempty(gridsize)
    if isempty(in)
        ha = gca;
    else
        fn_isfigurehandle(in);
        switch get(in,'type')
            case {'figure' 'uipanel'}
                ha = findobj(in,'type','axes');
                if ~isscalar(ha)
                    if strcmp(get(in,'type'),'figure'), clf(in), else delete(get(in,'children')), end
                    ha = axes('parent',in);
                end
            case 'axes'
                ha = in;
            otherwise
                error '''in'' argument is not an axes, figure or uipanel handle'
        end
    end                
    doxlabel = true;
    doylabel = true;
else
    ha = [];
    if isempty(in)
        hf = gcf;
        clf(hf)
    elseif ~isscalar(in)
        ha = in;
        hf = get(ha(1),'parent');        
        doxlabel = true(1,ngraph);
        doylabel = true(1,ngraph);
    else
        fn_isfigurehandle(in);
        hf = in;
        switch get(hf,'type')
            case 'figure'
                clf(hf)
            case 'uipanel'
                delete(get(hf,'children'))
            otherwise
                error '''in'' argument is not a figure or uipanel handle'
        end
    end
    
    if isempty(ha)
        if isempty(gridsize)
            ncol = ceil(sqrt(ngraph));
            nrow = ceil(ngraph/ncol);
        else
            nrow = gridsize(1);
            ncol = gridsize(2);
            if nrow*ncol<ngraph, error 'more graphs than grid cells', end
        end
        
        A = .06;
        B = .08;
        W = 1-A-.005;
        H = 1-B-.01;
        ww = W/ncol;
        hh = H/nrow;
        a = .002;
        b = .002;
        w = ww-a;
        h = hh-b-.002;
        
        ha = zeros(1,ngraph);
        doxlabel = false(1,ngraph);
        doylabel = false(1,ngraph);
        kgraph = 0;
        for i=1:nrow
            for j=1:ncol
                kgraph = kgraph+1;
                if kgraph>ngraph, break, end
                ha(kgraph) = axes('parent',hf,'pos',[A+(j-1)*ww+a B+(nrow-i)*hh+b w h]); %#ok<LAXES>
                doxlabel(kgraph) = (kgraph>ngraph-ncol);
                doylabel(kgraph) = (j==1);
            end
        end
    end
end
xx = double(calcium{1});
avgcalcium = mean(xx);
dodf = (avgcalcium(1)>.9 && avgcalcium(1)<1.1);
Flabel = fn_switch(dodf,'DF/F','F');

% Axis size and spike positions
if isempty(ylim)
    if isempty(xx)
        ylim = [0 1];
    else
        m = min(xx(:));
        M = max(xx(:));
        ylim = [m M] + (M-m)*[-.35 .1];
        if ~diff(ylim), ylim = ylim + [-.5 .5]; end
    end
end
spikeheight = .06*diff(ylim);
yspike = ylim(1)+spikeheight*[.75 3.25]; 
yspikelink = ylim(1)+spikeheight*[1.5 2.5];

% Colors, decorations, evaluation cost
if isfield(lineoptions,'color')
    color = lineoptions.color;
    if ischar(color) && isscalar(color)
        color = {color([1 1]) ['b' color([1 1])]};
    elseif ~iscell(color)
        error 'color argument'
    end
else
    color = {'rk' 'bkk'};
end
spikecol = color{1};
calciumcol = color{2};
switch displaymode
    case 'screen'
        calciumwidth = [1 2 2];
        labsz = 10;
        fsz = 8;
    case 'smallsym'
        calciumwidth = [1 2 2];
        labsz = 8;
        fsz = 6;
    case 'paper'
        calciumwidth = [.5 1 1];
        labsz = 9;
        set(ha,'fontsize',7,'defaulttextfontsize',9)
        fsz = 7;
    case 'factorbox'
        calciumwidth = [.5 1 1];
        labsz = 7;
        set(ha,'fontsize',5,'defaulttextfontsize',7,'box','on')
        fsz = 5;
end
haisreplace = strcmp(get(ha,'nextplot'),'replace');
set(ha(haisreplace),'nextPlot','replacechildren')
if isfield(lineoptions,'linewidth')
    calciumwidth = lineoptions.linewidth;
end
if isfield(lineoptions,'fullline')
    lineoptions.fullline = logical(lineoptions.fullline);
    calciumline = cell(1,length(lineoptions.fullline));
    [calciumline{lineoptions.fullline}] = deal('-');
    [calciumline{~lineoptions.fullline}] = deal('--');
else
    calciumline = {'-' '-' '--'};
end
nlinedef = min([length(calciumcol) length(calciumwidth) length(calciumline)]);
decmisses = 'xx'; %'x*';
decfalsep = 'oo'; %'^o';
switch displaymode
    case 'screen'
        decsiz = [6 6; 4 4];
        spkwidth = 1;
    case 'smallsym'
        decsiz = [4 4; 3 3];
        spkwidth = 1;
    case 'paper'
        decsiz = [3 3; 2 2];
        spkwidth = .7;
    case 'factorbox'
        decsiz = [3 3; 2 2];
        spkwidth = .5;
end
if isempty(cost)
    cost = spk_distance('par');
    cost.maxdelay = 1;
    doaddsides = true;
else
    doaddsides = false;
end

% Display
[stats.d stats.true stats.falsen stats.detections stats.falsed stats.miss stats.fd] = deal(zeros(1,ngraph));
okstat = false;
for k=1:ngraph
    
    spikesk = spikes{k};
    calciumk = calcium{k};
    
    % display calcium traces
    if isempty(calciumk)
        T = max(fn_map(@max,spikesk,'array'));
        if isempty(T), T = 1; end
        tt = 0:dt(k):T;
    else
        nt = size(calciumk,1);
        tt = (0:nt-1)*dt(k);
    end
    if ~isempty(calciumk)
        hl(k,:) = plot(tt,calciumk,'parent',ha(k)); %#ok<AGROW>
        for i=1:min(nlinedef,size(calciumk,2))
            set(hl(k,i),'color',calciumcol(i),'linewidth',calciumwidth(i),'linestyle',calciumline{i})
        end
    else
        cla(ha(k))
    end
    
    % display calcium events
    if ~isempty(calciumevents)
        ev = calciumevents(k); ne = length(ev.time);
        ck = calciumk(:,1);
        for i=1:ne
            tpos = ev.time(i) + [-1 1]*.2;
            ypos = max(ck(tt>tpos(1) & tt<tpos(2)));
            fn_arrow(ev.time(i)*[1 1],ypos+spikeheight*[1 .2],'40%',40,.4, ...
                'patch','color','m')
            if strcmp(calciumeventsmode,'full')
                str = sprintf('%.0f%% [%i]',ev.amp(i)*100,ev.num(i));
                text(ev.time(i),ypos+spikeheight*1.1,str, ...
                    'rotation',90,'verticalalignment','middle','horizontalalignment','left', ...
                    'fontsize',fsz,'fontweight','bold','color','m')
            end
        end
    end

    % display spikes
    for i=1:length(spikesk)
        if dorate && i>1, break, end
        fn_spikedisplay(spikesk{i},yspike(i),spikeheight,'parent',ha(k), ...
            'color',spikecol(i),'linewidth',spkwidth)
    end
    if length(spikesk)==2 && ~dorate
        okstat = true;
        if doaddsides, cost.sides = [0 tt(end)]; end
        [stats.d(k) desc] = spk_distance(spikesk{1},spikesk{2},cost);
        stats.true(k) = desc.count.true;
        stats.falsen(k) = desc.count.falsen;
        stats.detections(k) = desc.count.detections;
        stats.falsed(k) = desc.count.falsed;
        stats.miss(k) = stats.falsen(k)/stats.true(k);
        stats.fd(k) = stats.falsed(k)/stats.detections(k);
        stats.ER(k) = f1score(stats.miss(k),stats.fd(k));
        
        fn_spikedisplay(spikesk{1}(desc.miss0),yspike(1),'parent',ha(k), ...
            'marker',decfalsep(1),'markersize',decsiz(2,1),'linewidth',spkwidth,'color',spikecol(1))
        fn_spikedisplay(spikesk{1}(desc.miss1),yspike(1),'parent',ha(k), ...
            'marker',decfalsep(2),'markersize',decsiz(2,2),'linewidth',spkwidth,'color',spikecol(1))
        
        fn_spikedisplay(spikesk{2}(desc.falsep0),yspike(2),'parent',ha(k), ...
            'marker',decmisses(1),'markersize',decsiz(1,1),'linewidth',spkwidth,'color',spikecol(2))
        fn_spikedisplay(spikesk{2}(desc.falsep1),yspike(2),'parent',ha(k), ...
            'marker',decmisses(2),'markersize',decsiz(1,2),'linewidth',spkwidth,'color',spikecol(2))
        
        for i=1:size(desc.perfectshift,2)
            line([spikesk{1}(desc.perfectshift(1,i)) spikesk{2}(desc.perfectshift(2,i))],yspikelink, ...
                'parent',ha(k),'color',[.5 .5 .5],'linewidth',spkwidth)
        end
        for i=1:size(desc.okshift,2)
            line([spikesk{1}(desc.okshift(1,i)) spikesk{2}(desc.okshift(2,i))],yspikelink, ...
                'parent',ha(k),'color',[.5 .5 .5],'linewidth',spkwidth)
        end

        if strcmp(statflag,'local')
            %             str = { ... ['distance between spike trains: ' num2str(sum(stats.d),'%.2f')] ...
            %                 sprintf('misses: %.1f%%',stats.miss(k)/stats.real(k)*100), ...
            %                 sprintf('false positives: %.1f%%',stats.fp(k)/stats.detect(k)*100)};
            str = sprintf('ER: %.1f%% (miss: %.1f%% / fd: %.1f%%)',stats.ER(k)*100,stats.miss(k)*100,stats.fd(k)*100);
            if ~isempty(titl), str = [titl(k) str]; end %#ok<AGROW>
            text('parent',ha(k),'string',str,'units','normalized','pos',[.05 .95]);
        end
    elseif dorate
        if length(spikesk)~=2, error 'rate display only with 2 spike trains', end
        idx = (spikesk{2}>1e-3);
        y0 = yspike(2)-spikeheight/2;
        if any(idx)
            curhold = ishold(ha(k));
            hold(ha(k),'on')
            stem(tt(idx),y0+spikesk{2}(idx)*spikeheight,'parent',ha(k), ...
                'color',spikecol(2),'marker','none','basevalue',y0,'showbaseline','off');
            if ~curhold, hold(ha(k),'off'), end
        end
    end
    
    % auto-detect burst and display number of spikes per burst
    if isempty(burstdelay), burstdelay = .3; end
    spk = spikesk{1};
    nsp = fn_switch(dorate,1,length(spikesk));
    dtext = 0; %fn_coordinates(ha(k),'b2a',[1 0],'vector'); dtext = 4*dtext(1);
    for i=2:nsp, spk = union(spk,spikesk{i}); end
    if ~isempty(spk)
        delays = diff(spk);
        kburst = [1 1+find(delays>burstdelay)];
        nburst = length(kburst);
        burstend = [kburst(2:end)-1 length(spk)];
        tsep = [-Inf (spk(burstend(1:end-1))+spk(kburst(2:end)))/2 Inf];
        count = zeros(1,nsp);
        for kb=1:nburst
            tk = spk(burstend(kb));
            for i=1:nsp
                count(i) = sum(spikesk{i}>tsep(kb) & spikesk{i}<=tsep(kb+1));
            end
            if any(count>1)
                for i=find(count>1)
                    text(tk+dtext,yspike(i),[' ' num2str(count(i))], ...
                        'parent',ha(k),'color',spikecol(i),'fontsize',fsz, ...
                        'verticalalignment','middle')
                end
            end
        end
    end
    
    % axes decoration, more formatting
    axis(ha(k),[tt([1 end]) ylim])
    if doxlabel(k), xlabel(ha(k),'time (s)','fontsize',labsz), else set(ha(k),'xticklabel',''), end
    if doylabel(k), ylabel(ha(k),Flabel,'fontsize',labsz),     else set(ha(k),'yticklabel',''), end
    set(ha(haisreplace),'nextPlot','replacechildren')
    if strcmp(displaymode,'factorbox')
        % graph not visible, but lines to help reading
        set(ha,'visible','off')
        ylstep = fn_switch(ylim(2)<2,.1,ylim(2)<5,.5,1);
        yl1 = min(1,fn_round(ylim(1),ylstep,'ceil'));
        yl2 = fn_round(ylim(2),ylstep,'floor');
        yll = yl1:ylstep:yl2;
        hl = fn_lines('y',yll,'color',[1 1 1]*.6,'linestyle',':');
        set(hl(yll==1),'linestyle','-')
        for i=1:length(hl), uistack(hl(i),'bottom'), end
    end
    if ~isempty(toptitl), title(toptitl,'fontsize',labsz,'visible','on'), end
end

% Display global stats
if okstat && fn_ismemberstr(statflag,{'global' 'title'})
    %     fprintf('distance between spike trains: %.2f\n',sum(stats.d));
    %     fprintf('\b [misses: %.1f%%, false positives: %.1f%%]\n', ...
    %         sum(stats.miss)/sum(stats.real)*100,sum(stats.fp)/sum(stats.detect)*100);

    % print also the total misses and false positives inside the first graph
    %     str = { ... ['distance between spike trains: ' num2str(sum(stats.d),'%.2f')] ...
    %         sprintf('misses: %.1f%%',sum(stats.miss)/sum(stats.real)*100), ...
    %         sprintf('false positives: %.1f%%',sum(stats.fp)/sum(stats.detect)*100)};
    miss = sum(stats.falsen)/sum(stats.true);
    fp = sum(stats.falsed)/sum(stats.detections);
    str = sprintf('ER: %.1f%% (miss: %.1f%% / fd: %.1f%%)',f1score(miss,fp)*100,miss*100,fp*100);
    if ~isempty(titl), str = {titl str}; end
    switch statflag
        case 'global'
            text('parent',ha(1),'units','normalized','pos',[.05 .95],'string',str);
        case 'title'
            title(ha(1),str)
    end
end

% % Save figure automatically when in "factor box" mode
% if strcmp(displaymode,'factorbox')
%     fn_savefig(get(ha(1),'parent'),'autoname','pdf','scaling',1)
% end

% Output
if nargout==0
    clear stats
end

% avg = mean(calcium(:));
% if avg>.8 && avg<1.8, avg = 1; elseif min(calcium(:))<0 && max(calcium(:))>0, avg = 0; end
% 
% st = std(calcium(:));
% m = min(calcium(:));
% amp = max(calcium(:))-m;
% 
% a = min(avg-st/2,m-(avg-m)/10);
% d = fn_switch(avg,0,st/10,max(m/50,st/10));
% 
% fn_spikedisplay(spikes,[a a-d]*avg,'color','k')