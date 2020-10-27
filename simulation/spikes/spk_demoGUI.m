classdef spk_demoGUI < hgsetget
   
    properties
        sim     % simulation result
        res     % estimation result
    end
    properties (Dependent)
        pgen    % spikes parameters
        pcal    % calcium parameters
        pest    % estimation parameters
    end
    properties (Access='private')
        grob    % graphic objects
        Xgen    % control for spikes parameters
        Xcal    % control for calcium parameters
        Xest    % control for estimation parameters
        currentnlf
        currentnlb
        warningshown = false;
    end
    
    % Init
    methods
        function G = spk_demoGUI
            % graphic objects
            % (controls)
            hf = fn_figure('MLspike demo - Controls');
            fn_setfigsize(hf,700,600)
            set(hf,'numbertitle','off')
            G.grob.hf = hf;
            G.grob.pgen = uipanel('parent',hf,'pos',[.01 .01 .32 .98]);     % controls for spikes parameters
            G.grob.pcal = uipanel('parent',hf,'pos',[.34 .01 .32 .98]);     % controls for calcium parameters
            G.grob.pest = uipanel('parent',hf,'pos',[.67 .09 .32 .90]);     % controls for estimation parameters
            G.grob.uerase = uicontrol('parent',hf,'units','normalized','pos',[.67 .01 .32 .07], ...
                'string','Erase estimation result','callback',@(u,e)eraseresult(G));
            initcontrols(G)
            % (display)
            G.grob.display = fn_figure('MLspike demo - Result');
            set(G.grob.display,'numbertitle','off')
            % run
            simulspikes(G)
        end
        function initcontrols(G)
            
            % SPIKES SIMULATION
            pgen = struct('noiseseed',[], ...
                'rate',0.75,'T',60,'mode','bursty');
            spec = struct( ...
                'SPIKES__SIMULATION',        'label', ...
                'noiseseed','xstepper 1 0 1000', ...
                'rate',     'double', ...
                'T',        'double', ...
                'mode',     {{'fix-rate' 'vary-rate' 'bursty'}});
            G.Xgen = fn_control(pgen,spec,@(s)simulspikes(G),G.grob.pgen);
           
            % CALCIUM SIMULATION
            pcal = struct('dt',.02,'T',60, ...
                'noiseseed',[], ...
                'a',.1,'tau',1,'ton',0, ...
                'nonlinearity','saturation (OGB)', ...
                'saturation',.1,'hill',1.7, ...
                'p2',0.5,'p3',0.01, ...
                'sigma',.03, ...
                'drift__amp',.05,'drift__n',5);
            spec = struct( ...
                'CALCIUM__SIMULATION','label', ...
                'noiseseed','xstepper 1 0 1000', ...
                'dt','double','T','double', ...
                'a','double','tau','double','ton','double', ...
                'nonlinearity',{{'none' 'saturation (OGB)' 'cubic polynom (GCaMP)' 'Hill+sat. (GCaMP)'}}, ...
                'saturation','double','hill','double','p2','double','p3','double', ...
                'sigma','double', ...
                'drift__amp', 'double', 'drift__n',  'double');
            G.Xcal = fn_control(pcal,spec,@(s)simulcalcium(G),G.grob.pcal);
            G.currentnlf = pcal.nonlinearity;
            checkavailablepar(G,'cal');
            
            % ESTIMATION
            p = struct('estimate','MAP spike train', ...
                'a',pcal.a,'tau',pcal.tau,'ton',pcal.ton, ...
                'nonlinearity',pcal.nonlinearity, ...
                'saturation',pcal.saturation,'hill',pcal.hill, ...
                'p2',pcal.p2,'p3',pcal.p3, ...
                'sigma',.025,'drift',.015, ...
                'cmax',10,'nc',50,'np',50,'bmin',.7,'bmax',1.3,'nb',50);
            spec = struct( ...
                'ESTIMATION',   'label', ...
                'estimate',{{'MAP spike train' 'spike probabilities' 'spike samples'}}, ...
                'a','double','tau','double','ton','double', ...
                'nonlinearity',{{'none' 'saturation (OGB)' 'cubic polynom (GCaMP)' 'Hill+sat. (GCaMP)'}}, ...
                'saturation','double','hill','double','p2','double','p3','double', ...
                'sigma','double','drift','double', ...
                'below__are__discretization__parameters','label', ...
                'cmax','double','nc','double','np','double',...
                'bmin','double','bmax','double','nb','double');
            G.Xest = fn_control(p,spec,@(s)doest(G),G.grob.pest);
            checkavailablepar(G,'est');
            
            % no action if Matlab is busy
            set(findobj(G.grob.hf,'type','uicontrol','style','pushbutton'),'BusyAction','cancel')
            % normalized positions
            set(findobj(G.grob.hf,'type','uicontrol'),'units','normalized')
        end
    end
    
    % Get parameters
    methods
        function pgen = get.pgen(G)
            pgen = G.Xgen.s;
        end
        function pcal = get.pcal(G)
            p = G.Xcal.s;
            pcal = spk_calcium('par');
            pcal.dt = p.dt;
            pcal.T = p.T;
            pcal.a = p.a;
            pcal.tau = p.tau;
            pcal.ton = p.ton;
            pcal.saturation = p.saturation;
            pcal.hill = p.hill;
            pcal.pnonlin = [p.p2 p.p3];
            checkavailablepar(G,'cal');
            switch p.nonlinearity
                case 'none'
                    [pcal.saturation pcal.hill pcal.pnonlin] = deal(0,1,[]);
                case 'saturation (OGB)'
                    [pcal.hill pcal.pnonlin] = deal(1,[]);
                case 'cubic polynom (GCaMP)'
                    [pcal.saturation pcal.hill] = deal(0,1);
                case 'Hill+sat. (GCaMP)'
                    pcal.pnonlin = [];
            end
            pcal.sigma = p.sigma;
            pcal.drift.parameter = [p.drift__n p.drift__amp];
        end
        function pest = get.pest(G)
            p = G.Xest.s;
            % warning upon the introduction of a rise time
            if p.ton>0 && ~G.warningshown
                waitfor(warndlg({'You introduced a rising time in the estimation, this will raise the dimension of the state space from 2 to 3 and might slow down estimations.' ...
                    'Estimation speed can be improved by adjusting the "discretization parameters" so as to make a coarser discretization grid.' ...
                    'However, this needs to be done carefully in order not to degrade estimation accuracy. To do so, monitor the displays in the "tps_mlspike GRAPH SUMMARY" figure.' ...
                    'As a suggestion, some discretization parameters will now be automatically changed.'}))
                G.Xest.cmax = 5;
                G.Xest.nc = 45;
                G.Xest.bmin = 0.8;
                G.Xest.bmax = 1.2;
                G.Xest.nb = 35;
                G.warningshown = true;
                p = G.Xest.s;
            end
            % default values
            pest = tps_mlspikes('par');
            % regular parameters
            pest.algo.estimate = fn_switch(p.estimate, ...
                'MAP spike train','MAP','spike probabilities','proba','spike samples','samples');
            if strcmp(pest.algo.estimate,'samples')
                pest.algo.nsample = 4;
            end
            pest.dt = G.pcal.dt;
            pest.a = p.a;
            pest.tau = p.tau;
            pest.ton = p.ton;
            pest.saturation = p.saturation;
            pest.hill = p.hill;
            pest.pnonlin = [p.p2 p.p3];
            checkavailablepar(G,'est');
            switch p.nonlinearity
                case 'none'
                    [pest.saturation pest.hill pest.pnonlin] = deal(0,1,[]);
                case 'saturation (OGB)'
                    [pest.hill pest.pnonlin] = deal(1,[]);
                case 'cubic polynom (GCaMP)'
                    [pest.saturation pest.hill] = deal(0,1);
                case 'Hill+sat. (GCaMP)'
                    pest.pnonlin = [];
            end
            pest.finetune.sigma = p.sigma;
            pest.drift.parameter = p.drift;
            % discretization parameters
            pest.algo.cmax = p.cmax;
            pest.algo.nc = p.nc;
            pest.algo.np = p.np;
            pest.F0 = [p.bmin p.bmax];
            pest.algo.nb = p.nb;
        end
        function checkavailablepar(G,flag)
            switch flag
                case 'cal'
                    X = G.Xcal;
                case 'est'
                    X = G.Xest;
            end
            xx = X.controls;
            names = {xx.name};
            % nonlinearity parameters
            switch X.nonlinearity
                case 'none'
                    en = [0 0 0 0];
                case 'saturation (OGB)'
                    en = [1 0 0 0];
                case 'cubic polynom (GCaMP)'
                    en = [0 0 1 1];
                case 'Hill+sat. (GCaMP)'
                    en = [1 1 0 0];
            end
            F = {'saturation' 'hill' 'p2' 'p3'};
            for i=1:length(F)
                xi = xx(strcmp(names,F{i}));
                set([xi.hname xi.hval],'enable',fn_switch(en(i)))
            end
            % number of drifts only if drift amplitude
            if strcmp(flag,'cal')
                xi = xx(strcmp(names,'drift__n'));
                set([xi.hname xi.hval],'enable',fn_switch(X.drift__amp>0))
            end
            % discretization parameters that are really needed
            if strcmp(flag,'est')
                xi = xx(strcmp(names,'nb'));
                set([xi.hname xi.hval],'enable',fn_switch(X.drift>0))
                xi = xx(strcmp(names,'np'));
                set([xi.hname xi.hval],'enable',fn_switch(X.ton>0))
            end
        end
    end
    
    % Action
    methods
        function simulspikes(G)
            p = G.pgen;
            if ~isempty(G.Xgen.noiseseed), rng(G.Xgen.noiseseed,'twister'), end
            G.sim.spikesadd = spk_gentrain(p.rate,10+p.T,p.mode); % generate spikes before time 0 to get non-baseline calcium at time 0
            G.sim.spikes = G.sim.spikesadd(G.sim.spikesadd>10)-10;
            dt = G.pcal.dt;
            % display
            clf(G.grob.display)
            spk_display(dt,G.sim.spikes,[],'in',G.grob.display)
            % clear previous calcium
            G.sim.calcium = [];
            % proceed with calcium simulation?
            if G.Xcal.immediateupdate, simulcalcium(G), end
        end
        function simulcalcium(G)
            p = G.pcal;
            if ~isempty(G.Xcal.noiseseed), rng(G.Xcal.noiseseed,'twister'), end
            %if ismember('nonlinearity',G.Xcal.changedfields) && ~ismember('nonlinearpar',G.Xcal.changedfields), return, end
            p.T = 10+p.T;
            calciumadd = spk_calcium(G.sim.spikesadd,p); % generate calcium starting 10s before time 0
            G.sim.calcium = calciumadd(round(10/p.dt)+1:end); % keep only the part of the signal after time 0
            % display
            clf(G.grob.display)
            spk_display(p.dt,G.sim.spikes,G.sim.calcium,'in',G.grob.display)
            % also estimate?
            if G.Xest.immediateupdate, doest(G), end
        end
        function doest(G)
            p = G.pest;
            
            % estimate
            fn_watch(G.grob.hf,'startnow')
            [G.res.spikest G.res.fit G.res.drift] = spk_est(G.sim.calcium,p);
            
            % display
            hf = G.grob.display; clf(hf)
            switch p.algo.estimate
                case 'MAP'
                    spk_display(p.dt,{G.sim.spikes G.res.spikest}, ...
                        {G.sim.calcium G.res.fit G.res.drift},'in',hf)
                case 'proba'
                    spk_display(p.dt,{G.sim.spikes G.res.spikest}, ...
                        {G.sim.calcium G.res.fit G.res.drift},'in',hf,'rate')
                case 'samples'
                    for i=1:4
                        ha = subplot(2,2,i,'parent',hf);
                        ksamp = round(1+(i-1)/(4-1)*(p.algo.nsample-1));
                        spk_display(p.dt,{G.sim.spikes G.res.spikest{ksamp}}, ...
                            {G.sim.calcium G.res.fit(:,ksamp) G.res.drift(:,ksamp)},'in',ha)
                        ylabel(ha,sprintf('sample %i/%i',ksamp,p.algo.nsample))
                    end
            end
            fn_watch(G.grob.hf,'stop')
        end
        function eraseresult(G)
            clf(G.grob.display)
            spk_display(G.pcal.dt,G.sim.spikes,G.sim.calcium,'in',G.grob.display)
        end
    end
    
end
