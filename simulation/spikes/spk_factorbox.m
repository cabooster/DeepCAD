

%% FACTOR BOX
% This document illustrates qualitatively how the performance of MLspike
% depends on various primary and secondary factors.
% 
% The simulation on the right is characterized by a spiking rate of 1Hz,
% low photonic noise (RMS = 0.01) and small baseline drift/fluctuations. It
% serves as “starting point”. In the following simulations, various factors
% will be varied, one at a time, whereas the other simulation parameters
% stay unchanged (unless specified otherwise).


%% 0 Starting point: "standard" simulation
% The simulation we start from is characterized by low levels of photonic noise and drift-fluctuations and a spiking rate of 1Hz

% spikes
seed = 0;
T = 30; % time
dt = .02;
nt = T/dt;
tt = (0:nt-1)*dt;
rate = 1;
rng(seed,'twister')
spikes = spk_gentrain(rate,T);
spikespad = [spk_gentrain(rate,T) spikes+T];

% calcium with no noise
pcal = spk_calcium('par',dt,'T',2*T,'sigma',.01,'saturation',.1, ...
    'drift.parameter',[5 .015]);
rng(seed,'twister');
calciumpad = spk_calcium(spikespad,pcal); calcium = calciumpad(nt+1:2*nt);

% estimation
par = tps_mlspikes('par',dt,'saturation',.1, ...
    'finetune.sigma',.01,'drift.parameter',.01, ...
    'algo.interpmode','linear', ... % note that 'spline' would probably give better results
    'display','none','dographsummary',false);



[spikest fit drift] = spk_est(calcium,par);

%%
% display
clf reset
set(gcf,'color','w')
fn_setfigsize(gcf,[1200,220])
subplot(121)
ylim = [.8 1.3];
spk_display(dt,{spikes spikest},{calcium fit drift},ylim,'factorbox')
set(gca,'visible','on'), xlabel('time (s)','fontsize',6)

%% I Effect of primary factors on MLspike estimation accuracy

%% I/1) Noise level
%% I/1.1) photonic noise
% Increasing *photonic noise* results in poorer performance. However, it is
% not the high frequency part of this noise that hampers estimation. Rather,
% the critical part of the noise is around 1Hz, as shown further below.

ylim = [.6 1.4];

pcal1 = pcal; pcal1.sigma = .05;
rng(seed,'twister');
calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt+1:2*nt);
par1 = par; par1.finetune.sigma = .05;
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(121)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
    'toptitle',sprintf('(RMS=%g)',pcal1.sigma))

pcal1 = pcal; pcal1.sigma = .1;
rng(seed,'twister');
calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt+1:2*nt);
par1 = par; par1.finetune.sigma = .08;
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(122)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
    'toptitle',sprintf('(RMS=%g)',pcal1.sigma))

%% I/1.2) slow drifts
% Performance is quite robust against *low frequency noises* up to ~0.1Hz (this
% advantage is somewhat lost at spiking rates above ~10Hz).
desired_rms = .3;
rng(seed,'twister');
noise = simul_noiseequalizer(nt,dt,[.001 .1],1);
noise = noise * desired_rms/rms(noise);
calcium1 = calcium+noise;
par1 = par; par1.drift.parameter = .02;
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(121)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},'factorbox', ...
    'toptitle',sprintf('(<0.1Hz, RMS=%g)',desired_rms))

desired_rms = .5;
rng(seed,'twister');
noise = simul_noiseequalizer(nt,dt,[.001 .1],1);
noise = noise * desired_rms/rms(noise);
calcium1 = calcium+noise;
par1 = par; par1.drift.parameter = .03;
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(122)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},'factorbox', ...
    'toptitle',sprintf('(<0.1Hz, RMS=%g)',desired_rms))

%% I/1.3) worst-band noise
% Noises in the *"worst band" (0.1-3Hz)* most strongly impact on the estimation's accuracy.
desired_rms = .015;
rng(seed,'twister');
badnoise = simul_noiseequalizer(nt,dt,[.1 3],1);
badnoise = badnoise * desired_rms/rms(badnoise);
calcium1 = calcium+badnoise;
par1 = par; par1.finetune.sigma = .03;
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(121)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
    'toptitle',sprintf('(RMS=%g)',desired_rms))
drawnow

desired_rms = .05;
rng(seed,'twister');
badnoise = simul_noiseequalizer(nt,dt,[.1 3],1);
badnoise = badnoise * desired_rms/rms(badnoise);
calcium1 = calcium+badnoise;
par1 = par; par1.finetune.sigma = .05;
par1.algo.interpmode = 'spline'; % slight improvement in the estimation, not clear in general which of 'linear' or 'spline' is better
[spikest1 fit1 drift1] = spk_est(calcium1,par1);
subplot(122)
spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
    'toptitle',sprintf('(RMS=%g)',desired_rms))

%% I/2) Spiking rate and spiking regularity
% Higher spiking rates result in poorer estimations, mainly because it
% becomes more difficult to correctly estimating the baseline. Therefore,
% the spiking pattern is also important: bursts alternating with silent
% periods eases baseline estimation (first row), wheras near to regularly
% dense spiking (last row) impairs it. Note that noise also hampers
% baseline estimation: in the bottom-left example, baseline and spikes are
% exquisitely recovered but when noise hides the individual spike onsets,
% estimation quality rapidly deteriorates.

figure(gcf)
fn_setfigsize(gcf,[1200,600])

rates = [5 20];
type = {'vary-rate' 'fix-rate' 'periodic'};
sigma1 = .01;
plotnum = 320;

for j = 1:length(type)
        for i = 1:length(rates)
            % spike train with different rate
            ratei = rates(i);
            if strcmp(type(j),'vary-rate')
                % vary-rate has period with no spikes and other periods
                % with high rate: we specify a lower "average rate" so that
                % the periods with high rate will have approximately the
                % same spiking rate as in the fix-rate case
                ratei = ratei * .7;
            end
            rng(seed,'twister');
            spikes1 = spk_gentrain(ratei,T,type{j});
            spikespad1 = [spk_gentrain(rates(i),T,type{j}) spikes1+T];
            
            % calcium
            pcal1 = spk_calcium('par',dt,'T',2*T,'sigma',sigma1,'saturation',.1, ...
                'drift.parameter',[5 .015]);
            rng(seed,'twister');
            calciumpad1 = spk_calcium(spikespad1,pcal1);
            calcium1 = calciumpad1(nt+1:2*nt);
            
            % estimation
            par1 = tps_mlspikes('par',dt,'saturation',.1, ...
                'finetune.sigma',sigma1,...
                'drift.parameter',.005,... % it helped to decrease this parameter, especially in the "vary-rate" case
                'display','none','dographsummary',false, ...
                'algo.cmax',60, ... % IT IS NECESARY TO INCREASE THE DEFAULT VALUE (INITIALLY 10)
                'algo.nc',500,'algo.interpmode','linear');
            par1.finetune.spikerate = rates(i);
            [spikest1 fit1 drift1] = spk_est(calcium1,par1);
            
            % display
            subplot(plotnum+2*(j-1)+i)
            ylim = [0.7 1.8];
            spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
                'toptitle',sprintf('(Spikerate=%g Hz)',par1.finetune.spikerate))
            ylabel(type{j},'fontsize',6,'visible','on')
            if j>1, title(''), end
            drawnow
        end
    end


%% II Effect of secondary factors
% A number of factors can be considered as "secondary", in the sense
% that their effect can be reduced to the effect of one
% or more of the primary factors described above.

%% II/1) Ca-fluorescence transient amplitude
% Decrease of unitary Ca transient amplitude (fed
% to the algorithm) has a similar effect as
% increasing noise: more misses occur, as it becomes more
% difficult to distinguish spikes from noise. Baseline estimation is also
% impaired.

figure(gcf)
fn_setfigsize(gcf,[1200,220])
ylim = [0.7 1.35];
plotnum = 120;
pcal1 = pcal;
amps = [0.1 0.025];
ntrial = length(amps);
for i = 1:ntrial
    pcal1.a = amps(i);
    rng(seed,'twister');
    calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt+1:2*nt);
    par1 = par; par1.finetune.sigma = .01;
    par1.a = amps(i);
    [spikest1 fit1 drift1] = spk_est(calcium1,par1);
    subplot(plotnum+i)
    spk_display(dt,{spikes spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
        'toptitle',sprintf('(amplitude of unitary Ca transient=%g)',pcal1.a))
end

%% II/2) increasing laser intensity
% Increasing laser intensity decreases the photonic noise in the ?F/F
% signals (by the square root of the increase in signal). This improves
% spike estimation significantly only if the noise in the 0.1-3Hz frequency
% band is of photonic origin, but not otherwise.


%% II/3) changing sampling rate at constant laser power
% Left vs. middle panel: When decreasing sampling rate by scanning the same
% field of view at lower speed, the SNR in individual frames increases (by
% the square root of the speeds' ratio, due to increased dwell time: here,
% a 4x slower scanning results in 2x higher SNR). Yet, the estimation
% quality remains similar (main text, Fig. 1c, right), because the noise
% level (quantified as RMS in the 0.1-3Hz frequency band) remains constant.
% However, at low noise levels, slow sampling may obviously reduce temporal
% precision (Supp. Fig. 1). 
% Left panel vs. right panel: When decreasing sampling rate by scanning a
% larger field of view without changing scanning speed, the SNR in
% individual frames does not increase (dwell time remains unchanged),
% reducing estimation accuracy.


figure(gcf), clf
fn_setfigsize(gcf,[1200,220])

dts = [.02 .08 .08];        % original sampling rate (50Hz) and 4x slower (200Hz)
%sigmas = [1 .5 1]*.04;    % "reasonable" noise SNR (.035) or 2x less (.0175)
sigmas = [1 .5 1]*.05;    % "reasonable" noise SNR (.035) or 2x less (.0175)
rng(seed,'twister');

% decrease the gap between the subplots
ha = [subplot(131) subplot(132) subplot(133)];
p1 = get(ha(1),'pos'); p3 = get(ha(3),'pos');
x0 = p1(1); x1 = p3(1)+p3(3); y0 = p1(2); h = p1(4);
gap = .03; w = (x1-x0-2*gap)/3;
fn_set(ha,'position',{[x0 y0 w h] [x0+w+gap y0 w h] [x0+2*w+2*gap y0 w h]})



for i = 1:length(dts)
    % calcium with no noise
    rng(seed,'twister');
    pcal1 = pcal;
    if i==2
        nt1 = T/dts(1);
        pcal1.dt  = dts(1);
        pcal1.sigma = sigmas(1);
        calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt1+1:2*nt1);
        calcium1 = fn_bin(calcium1,4);
    elseif i==3
        nt1 = T/dts(1);
        pcal1.dt  = dts(1);
        pcal1.sigma = sigmas(1);
        calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt1+1:2*nt1);
        calcium1 = calcium1(1:4:end);
    else
        nt1 = T/dts(i);
        pcal1.dt  = dts(i);
        pcal1.sigma = sigmas(i);
        calciumpadi = spk_calcium(spikespad,pcal1); calcium1 = calciumpadi(nt1+1:2*nt1);
    end
    par1 = par;
    par1.finetune.sigma = sigmas(i)*.65; % the *.65 results in having more equilibrated misses and fp.
    par1.dt = dts(i);
    [spikest1 fit1 drift1] = spk_est(calcium1,par1);
    axes(ha(i))
    if i==1
        titl = sprintf('(sr=%gHz, RMS=%g)',1/pcal1.dt,pcal1.sigma);
    elseif i==2
        titl = sprintf('(sr=%gHz, RMS=%g)',1/pcal1.dt/4,pcal1.sigma/2);
    else
        titl = sprintf('(sr=%gHz, RMS=%g)',1/pcal1.dt/4,pcal1.sigma);
    end
    spk_display(dts(i),{spikes spikest1},{calcium1 fit1 drift1},ylim, ...
        'factorbox','toptitle',titl)
    drawnow
end

%% II/4) contamination
% The  fluorescence recorded from one cell can be contaminated by the signal from another
% one (or by the signal from the neuropil, as can happen for example when
% imaging with low numerical aperture or from deep locations).
% Dealing with this kind of "noise" is particularly challenging.
% Nevertheless, the algorithm handles quite well up to more than 30%
% contamination, as shown below. Keeping the amplitude set to its
% "uncontaminated" rather than to its actual value further penalizes the contaminating Ca
% transients leading to improved estimations (middle vs. right panel).

figure(gcf), clf
fn_setfigsize(gcf,[1200,320])

% spiketrain of neuron 1
dt = .02;
nt = T/dt;
rate = 1;
ylim = [0.8 1.5];
par1 = par;
[spikes_c, spikespad_c, calciumpad_c, spikest_c, calcium_c, fit_c, drift_c] = deal(cell(1,2));
mixings = [1/9 3/7 3/7];
DOCHANGEA = [false false true];
plotnum = 130;

% signal of the 2 neurons
for i = 1:2
    seedi = i-1; % do not use variable 'seed', otherwise this will change the results of other simulations
    rng(seedi,'twister')
    spikes_c{i} = spk_gentrain(rate,T);
    spikespad_c{i} = [spk_gentrain(rate,T) spikes_c{i}+T];
    rng(seedi,'twister');
    pcal1 = spk_calcium('par',dt,'T',2*T,'sigma',.01,'saturation',.1, ...
        'drift.parameter',[5 .015]);
    calciumpad_c{i} = spk_calcium(spikespad_c{i},pcal1); calcium_c{i} = calciumpad_c{i}(nt+1:2*nt);
    %     [spikest_c{i} fit_c{i} drift_c{i}] = spk_est(calcium_c{i},par1);
    %     subplot(plotnum+i)
    %     spk_display(dt,{spikes_c{i} spikest_c{i}},{calcium_c{i} fit_c{i} drift_c{i}},ylim,'factorbox')
    %     title(sprintf('Ca signal%g and its estimation (no contamination)',i));
end

% contaminated signal
for i=1:length(mixings)
    w = mixings(i);
    f1 = 1/(w+1);
    f2 = w/(w+1);
    calcium1 = calcium_c{1}*f1 + calcium_c{2}*f2;
    spikes1 = spikes_c{1};
    par3 = par;
    if DOCHANGEA(i)
        par3.finetune.sigma = sqrt((f1*pcal1.sigma)^2 + (f2*pcal1.sigma)^2); %
        par3.a = f1*pcal1.a;
    end
    [spikest1 fit1 drift1] = spk_est(calcium1,par3);
    subplot(plotnum+i)
    titl = sprintf(['Signal = %g%% correct + %g%% contamination\n' ...
        'Algorithm expects %g%%\\DeltaF/F transients (%s)'], ...
        round(100/(w(1)+1)), round(100*w(1)/(w(1)+1)), ...
        par3.a*100,fn_switch(DOCHANGEA(i),'correct','too high'));
    spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim, ...
        'factorbox','toptitle',titl)
    hl = line(tt,(calcium_c{1}-1)*f1+1.35); uistack(hl,'bottom')
    hl = line(tt,(calcium_c{2}-1)*f2+1.25); uistack(hl,'bottom')
    set(gca,'ylim',[.8 1.6])
end

%% II/5) numerical aperture 
    % 
    % Lower numerical apertures result in lower SNR, because of  more photonic noise (black) 
    % and more contamination by other cells and/or
    % the neuropil (red). SNR was computed as: unitary peak response amplitude/ devided by
    % the standard deviation of the fluorescence baseline. SNR was then normalized dividing bythe value at maximal NA. Data stem from
    % whole-cell patched GCaMP6f neurons in mice cortical slices recorded
    % with a variable NA.
    % Points and errorbars represent, respectively, mean and std of 10
    % unitary fluorescence transients.
    
    
    % Vectors of NAs, SNRs and SNR compensated loss at aperture (same laser
    % power under objective) and their std over 10 APs.
    navector=[1.00 0.74 0.51 0.32 0.20];
    rmsv=[1.00 1.03 0.60 0.33 0.17];
    stdv=[0.0088 0.0105 0.0128 0.0139 0.0134]/35.947;
    rmsc=[1.00 1.03 0.58 0.55 0.33];
    stdc=[0.0088 0.01 0.0119 0.0126 0.0131]/35.947;

    figure(gcf), clf, fn_setfigsize(gcf,540,433)
    yy=errorbar(navector,rmsv,stdv,'k-');set(yy,'linewidth',2);
    hold on
    zz=errorbar(navector,rmsc,stdc,'r-');set(zz,'linewidth',2);
    hold off 
    
    ww=ylabel('SNR (norm.)');
    ww=xlabel('NA');
    ww=title('SNR as a function of numerical aperture');
    ww=legend([yy;zz],'raw', 'compensated for power loss at aperture');
    set((gca),'xdir','reverse')
    set(gca,'xtick',fliplr(navector))
    drawnow
    
%% II/6) recording depth 
% The effects of recording depth can vary importantly between different
% experimental conditions and depend on on tissue transparency, the
% efficacy of scattered fluorescence collection, and the locality and
% sparseness of the Ca probe’s distribution. In general, recording deep
% inside scattering tissues reduces the signal (less ballistic excitation
% photons reach the sample), thus demanding high laser power, and increases
% contamination by other cells and/or the neuropil. This contamination
% originates both from the neighborhood of the recorded cell (due to a
% larger point spread function of the focused beam), and from the
% superficial layers where the probe is excited despite a non-focused laser
% beam, because the latter is strong and only weakly attenuated by tissue
% scattering (see simulations below). In order to avoid contamination from
% surface fluorescence upon deep imaging, it is thus preferable to load
% only the imaged area with the calcium sensor, rather than the full
% volume.

% Simulation of the laser beam focused deep inside the tissue
xx = -800:.2:800;
zz = 0:.2:600;
[xxx zzz] = ndgrid(xx,zz);
halflen = 100;
z1 = 5*halflen; % 2*halflen
xres = 1;
zres = 5;
theta = 45;
zatten = exp(-zz/halflen);
xspread = exp(-xxx.^2./(2*(xres^2+(tand(theta)*(zzz-z1)).^2)));
r = .1+repmat(abs(xx(:)),[1 length(zz)]);
xspread2d = r.*xspread;
tmp = sum(xspread2d,1);
xspread = fn_div(xspread,tmp);
xspread2d = fn_div(xspread2d,tmp);
F = fn_mult(zatten,xspread).^2;

% Display image of beam focusing inside tissue
fn_setfigsize(gcf,[1200,220])
colormap(jet(256))

subplot(1,4,1:3)
imagesc(xx,zz,log10(F'/max(F(:))),[-8 0])
axis image
set(gca,'fontsize',7)
line(xx([1 end]),[1 1]*halflen,'color','w','linestyle','--')
line(xx([1 end]),[1 1]*z1,'color','w','linestyle','--')
xlabel 'x (\mum')
ylabel 'depth (\mum)'

colorbar

% Display sum of all fluorescence from each depth
subplot(1,4,4)
zspreadint = sum(r.*F,1);
plot(zspreadint/max(zspreadint),zz)
set(gca,'ydir','reverse')
disp(sprintf('%.0f%% of the signal at the focus',sum(zspreadint(zz>z1-50 & zz<z1+50))/sum(zspreadint)*100))
set(gca,'fontsize',7)

% free memory
clear par3 spikes_c spikespad_c calcium_c calciumpad_c fit_c drift_c

%% II/7) calcium sensors
% Carefully exploring all factors making some sensors preferable over
% others in general (ability to do chronic imaging, target specific cells,
% pharmacological side-effects, etc.) goes beyond the scope of this work.
% Here, we characterize only those that directly affect the ability to
% estimate spikes from calcium recordings. Their transient amplitude for
% one spike directly influences the SNR; their rise and decay times
% influence the temporal precision of the estimations and the ability to
% follow high spiking rates – which can also be limited by saturation;
% finally, the a priori knowledge on the exact values of these parameters
% and their variability also influence the ability to perform
% autocalibration: e.g., we still miss knowledge on the exact function
% governing nonlinearities of GCaMP6 sensors.
%
% In the graphs below we compare spike estimation accuracy on  data
% simulated using characteristics of GCaMP6s and GCaMP6f. At low spiking
% rate (left column), GCaMP6s clearly outperforms GCaMP6f thanks to its
% larger unitary fluorescence transients: it can accomodate a noise 5 times
% larger at comparable level of accuracy (note that OGB dye would be
% positioned halfway). This advantage is progressively lost at higher
% spiking rates (top left). Finally, only GCaMP6f can follow
% (Poisson-statistics) trains of spikes at 20sp/s (bottom right), while
% GCaMP6s is limited to about 5sp/s.

pcalG = spk_calcium('par',dt,'T',2*T,'sigma',.01,'saturation',.1, ...
    'drift.parameter',[5 .015]);
% parG = tps_mlspikes('par',dt,'saturation',.1, ...
%     'finetune.sigma',.01,'drift.parameter',.01, ...
%     'display','none','dographsummary',true, ...
%     'F0',[.8 1.2],'algo.cmax',10,'algo.nc',50,'algo.np',50,'algo.nb',40,'algo.interpmode','linear');
parG = tps_mlspikes('par',dt,'saturation',.1, ...
    'finetune.sigma',.01,'drift.parameter',.01, ...
    'display','none','dographsummary',false, ...
    'F0',[.8 1.2],'algo.cmax',10,'algo.nc',120,'algo.np',50,'algo.nb',40,'algo.interpmode','linear');
pcal6s=pcalG;
par6s=parG;
pcal6f=pcalG;
par6f=parG;

% the set values are those from Cheng et al, 2013. the commented ones are
% those from Thomas with Svoboda-like normalization.
pcal6s.tau = 0.786; % 1.84;
pcal6s.a = 0.22; % 0.254;
pcal6s.ton = 0.190; % 0.0677;
pcal6s.saturation = 0.00629;
pcal6s.hill = 1.73;

pcal6f.tau = 0.214; % 0.716.
pcal6f.a = 0.12; % 0.092;
pcal6f.ton = 0.045; % 0.0156;
pcal6f.saturation = 0.00195;
pcal6f.hill = 2.05;

par6s.finetune.sigma = 0.01;
par6s.finetune.spikerate=0.1;
par6s.tau = 0.786; % 1.84;
par6s.a = 0.22; % 0.254;
par6s.ton = 0.190; % 0.0677;
par6s.saturation = 0.00629;
par6s.hill = 1.73;
par6s.drift.parameter = 0.002;

par6f.finetune.sigma = 0.01;
par6f.finetune.spikerate=0.1;
par6f.tau = 0.214; % 0.716;
par6f.a = 0.12; % 0.092;
par6f.ton = 0.045; % 0.0156;
par6f.saturation = 0.00195;
par6f.hill = 2.05;
par6f.drift.parameter = 0.002;

fn_setfigsize(gcf,[1200,400])

% spikes
seed = 0;
T = 30; % time
dt = .02;
nt = T/dt;
tt = (0:nt-1)*dt;
sigmanoise6s=[0.1467 0.0733]; % snr of 1.5 and 3
sigmanoise6f=[0.0267 0.12]; % snr of 4.5 and 1
% rate6s = [1 7];
rate6s = [1 5];
rate6f = [1 20];

savedir = tempdir;

for i=1:2 % loop over spiking rates
    for j = 1:2 % loop over the 2 GCaMPs
        % calcium with different noiselevels, depending on type of GCaMP6
        
        if j==1% GCaMP6s
            rate6 = rate6s(i);
            pcal6s.sigma = sigmanoise6s(i); %(good noise (ER=0%) for 6s: 0.125, that is a/sigma=RMS=1.76,
            % which yields sigma = 0.0682 for 6f, after rescaling for the smaller amplitude)
            pcal6=pcal6s;
            par6=par6s;
            type='s';
        elseif j==2 % GCaMP6f
            rate6 = rate6f(i);
            pcal6f.sigma = sigmanoise6f(i); % at 20 Hz a snr of 1 (rms=0.12) yields ER=2.1%)
            % spikerate of 5Hz yiels ER=0. At 1Hz, we don't get below ER=15% (set
            % par.finetune.sigma to 0.042)).
            pcal6=pcal6f;
            par6=par6f;
            type='f';
        end
        rng(seed,'twister')
        spikes1 = spk_gentrain(rate6,T,'fix-rate');
        spikespad1 = [spk_gentrain(rate6,T,'fix-rate') spikes1+T];
        calciumpad1 = spk_calcium(spikespad1,pcal6); calcium1 = calciumpad1(nt+1:2*nt);
        
        if (j==1) && (i==2)
            % 5Hz is quite a high spiking rate for 6s, minimize error by
            % using a very fine discretization (this slows down computation
            % even more however)
            par6.algo.nc=100;
            par6.algo.np=100;
            par6.algo.nb=100;
        end
        
        % estimation (store result so that it will not be needed to repeat
        % the same calculation in the future)
        H = fn_hash({calcium1,par6},8);
        fsave = fullfile(savedir,[H '.mat']);
        if exist(fsave,'file')
            [spikest fit drift] = fn_loadvar(fsave);
        else
            [spikest fit drift] = spk_est(calcium1,par6);
            fn_savevar(fsave,spikest,fit,drift)
        end
        
        % display
        subplot(2,2,i+2*j-2)
        if i==1
            ylim = [0 2];
        elseif i==2
            ylim = [-3 10];
        end
        spk_display(dt,{spikes1 spikest},{calcium1 fit drift},ylim,'factorbox');
        set(gca,'visible','on')
        ylabel(sprintf('(RMS=%g)',round(100*pcal6.sigma)/100),'fontsize',7);
        title(sprintf('GCaMP6%s; spiking rate = %gHz',type,rate6),'fontsize',7)
        drawnow
    end
end

%% III Fine-tuning of MLspike algorithm
% MLspike estimation accuracy obviously depends on its parameter settings.
% Here, we assume physiological parameters (A, tau, nonlinearity of the
% probe) to be known (see next section on autocalibration when this is not
% the case) and investigate the effects of the three remaining parameters:
% sigma, drift, and spikerate.
%
% These parameters code for the a priori level of expected photonic noise,
% slow drifts and spiking activity present in the data. Their relative
% settings therefore influence how the algorithm interprets ambiguous
% variations of the signal.

%% III/1) A priori noise level: parameter 'sigma'
% sigma is the expected RMS of a temporally uncorrelated noise. When the
% signals are unambiguous, because the noise is small (first line:
% RMS=0.01, 2 spikes/s), a wide range of sigma values leads to correct
% estimations. However, when the noise is large (second line: RMS=0.04),
% low values of sigma amount to "over-trusting the data " and cause false
% detections (as well as an underestimation of the baseline level:, left).
% Conversely, high sigma values amount to "not trusting the data enough",
% increasing the number of  misses (right).

figure(gcf), clf
fn_setfigsize(gcf,[1200,400])

% decrease the gap between the subplots
ha = [subplot(231) subplot(232) subplot(233); subplot(234) subplot(235) subplot(236)];
p1 = get(ha(1,1),'pos'); p3 = get(ha(1,3),'pos'); p4 = get(ha(2,1),'pos');
x0 = p1(1); x1 = p3(1)+p3(3); y0 = p1(2); h = p1(4); y1 = p4(2);
gap = .03; w = (x1-x0-2*gap)/3;
fn_set(column(ha),'position', ...
    column({[x0 y0 w h] [x0+w+gap y0 w h] [x0+2*w+2*gap y0 w h]; ...
    [x0 y1 w h] [x0+w+gap y1 w h] [x0+2*w+2*gap y1 w h]}))


sigmas = [.01 .04];    % "low" noise SNR or stronger noise
rates = [2 2];
nsigma = length(sigmas);
ylim = [.7 1.35];
%sigmatunefactors = {1 [] 0.5 2};
% sigmatunefactors = {1 2 .75};
% ntunesigma = length(sigmatunefactors);
sigmatunes = [.01; .02]*[.75 2 5];
ntunesigma = length(sigmatunes);
rng(seed,'twister');
for i = 1:nsigma
    rng(seed,'twister');
    ratei=rates(i);
    pcal1 = pcal;
    pcal1.sigma = sigmas(i);
    spikesi = spk_gentrain(ratei,T);
    spikespadi = [spk_gentrain(ratei,T) spikesi+T];
    rng(seed,'twister');
    calciumpadi = spk_calcium(spikespadi,pcal1); calcium1 = calciumpadi(nt+1:2*nt);
    par1 = par;
    par1.finetune.spikerate = 1; % default parameter value of 0.1 was too low (i.e. biased towards more misses)
    for k=1:ntunesigma
        %par1.finetune.sigma = sigmas(i)*sigmatunefactors{k};
        par1.finetune.sigma = sigmatunes(i,k);
        [spikest1 fit1 drift1 par2] = spk_est(calcium1,par1);
        axes(ha(i,k))
        spk_display(dt,{spikesi spikest1},{calcium1 fit1 drift1},ylim,'factorbox')
        if k==1, ylabel(sprintf('(RMS=%g)',pcal1.sigma),'fontsize',7,'visible','on'), end
        title(sprintf('(sigma=%g)',par2.finetune.sigma),'fontsize',7,'visible','on')
        drawnow
    end
end

%% III/1.1) Automatic estimation of parameter 'sigma'
% Parameter sigma can be autocalibrated from the data themselves. The
% simulations below show estimated sigma values and resulting spike
% estimations for the same signals as above (left and center).
% Autocalibration is possible also when the noise is temporally correlated
% (right).

figure(gcf), clf
fn_setfigsize(gcf,[1200,220])

% decrease the gap between the subplots
ha = [subplot(131) subplot(132) subplot(133)];
p1 = get(ha(1),'pos'); p3 = get(ha(3),'pos');
x0 = p1(1); x1 = p3(1)+p3(3); y0 = p1(2); h = .65;
gap = .03; w = (x1-x0-2*gap)/3;
fn_set(ha,'position',{[x0 y0 w h] [x0+w+gap y0 w h] [x0+2*w+2*gap y0 w h]})

ylim = [.7 1.35];
whitenoise = [true true false];
rmsi = [.01 .04 .04];
rate1 = 2;
ncol = length(whitenoise);
rng(seed,'twister')
spikes1 = spk_gentrain(rate1,T);
spikespad1 = [spk_gentrain(rate1,T) spikes1+T];
par1=par;
par1.finetune.spikerate = 1;
for i = 1:ncol
    pcal1 = pcal;
    rng(seed,'twister');
    if whitenoise(i) % the case of uncorrelated  (i.e. white) noise
        pcal1.sigma = rmsi(i);
        calciumpad1 = spk_calcium(spikespad1,pcal1);
        calcium1 = calciumpad1(nt+1:2*nt);
        ylab = sprintf('white noise (RMS=%g)',pcal1.sigma);
    else % correlated (i.e. band-passed) noise
        freqs = [.1 20]; % white noise will be filtered between these 2 frequencies
        %noise = simul_noiseequalizer(nt,dt,freqs,1);
        noise = fn_filt(randn(nt,1)*rmsi(i),1./fliplr(freqs)/dt,'b');
        pcal1.sigma = 0;
        rng(seed,'twister');
        calciumpad1 = spk_calcium(spikespad1,pcal1);
        calcium1 = calciumpad1(nt+1:2*nt) + noise;
        ylab = 'temporally correlated noise';
    end
    % estimating sigma using autocalibration
    sigmaest1 = spk_autosigma(calcium1,dt);
    par1.finetune.sigma = sigmaest1;
    [spikest1 fit1 drift1] = spk_est(calcium1,par1);
    axes(ha(i))
    spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim,'factorbox')
    ylabel(ylab)
    title(sprintf('(sigma_{est}.=%.2g)',par1.finetune.sigma),'fontsize',7,'visible','on')
    drawnow
end

%% III/2) A priori drift level: parameter 'drift'
% The drift parameter has a similar effect as sigma, but for low-frequency
% noises: Setting a low value results in fluctuations to be mistaken for
% spikes (second line, left), while setting a high value results in spikes
% to be mistaken for drifts/fluctuations (right). However, estimations are
% robust with respect to mis-setting the drift parameter (first line),
% provided the signals are not too ambiguous (second line).


figure(gcf), clf
fn_setfigsize(gcf,[1200,400])

% decrease the gap between the subplots
ha = [subplot(231) subplot(232) subplot(233); subplot(234) subplot(235) subplot(236)];
p1 = get(ha(1,1),'pos'); p3 = get(ha(1,3),'pos'); p4 = get(ha(2,1),'pos');
x0 = p1(1); x1 = p3(1)+p3(3); y0 = p1(2); h = p1(4); y1 = p4(2);
gap = .03; w = (x1-x0-2*gap)/3;
fn_set(column(ha),'position', ...
    column({[x0 y0 w h] [x0+w+gap y0 w h] [x0+2*w+2*gap y0 w h]; ...
    [x0 y1 w h] [x0+w+gap y1 w h] [x0+2*w+2*gap y1 w h]}))

rate1 = 1;
sigmas = [.01 .04];    % "low" noise SNR or stronger noise
nsigma = length(sigmas);
driftpar = [.003 .01 .03];
ntunedrift = length(driftpar);
for i = 1:nsigma
    pcal1 = pcal;
    pcal1.sigma = sigmas(i);
    pcal1.drift.parameter = [7 .02]; % make the drift a little "stronger"
    rng(seed,'twister');
    spikesi = spk_gentrain(rate1,T,'vary-rate',1); % using a non-constant spiking rate creates bursts that might be confused with baseline drifts
    spikespadi = [spk_gentrain(rate1,T) spikesi+T];
    rng(seed,'twister');
    calciumpadi = spk_calcium(spikespadi,pcal1); calcium1 = calciumpadi(nt+1:2*nt);
    par1 = par;
    par1.finetune.spikerate = 1; % default parameter value of 0.1 was too low (i.e. biased towards more misses)
    par1.finetune.sigma = sigmas(i);
    for k=1:ntunedrift
        par1.drift.parameter = driftpar(k);
        [spikest1 fit1 drift1] = spk_est(calcium1,par1);
        axes(ha(i,k))
        spk_display(dt,{spikesi spikest1},{calcium1 fit1 drift1},ylim,'factorbox')
        if k==1, ylabel(sprintf('(RMS=%g)',pcal1.sigma),'fontsize',7,'visible','on'), end
        title(sprintf('(a priori drift=%g)',par1.drift.parameter),'fontsize',7,'visible','on')
        drawnow
    end
end

%% III/3) A priori spiking rate: parameter 'spikerate'
% Increasing parameter spikerate increases the algorithm’s tendency to
% assign spikes (i.e. decrease misses but increase false detections). Note
% that the optimal value for spikerate is not necessarily the true spike
% rate. Note also that the 3 parameters sigma, drift and spikerate together
% control only 2 degrees of freedom of the estimation, as increasing one of
% them has exactly the same effect as decreasing the other two. In
% practice, parameter spikerate can thus be assigned a fix value (we
% usually use 1sp/s) while the other two parameters are fine tuned if
% necessary.


%% IV Autocalibration of physiological parameters
% Parameters A (the fluorescence transient amplitude for one spike) and ?
% (decay time constant) can be autocalibrated. The autocalibration
% algorithm (see Fig. 1e,f and Sup. Figs. 7,8) detects isolated calcium
% events and uses a histogram of all event amplitudes in order to assign a
% number of spikes to each event and to finally return estimated values for
% A and ?.

% autocalibration default parameters
pax0 = spk_autocalibration('par');
% set frame duration
pax0.dt = dt;
% set saturation
pax0.saturation = 0.1;
% set range
pax0.amin = 0.05;
pax0.amax = 0.2;
% change some parameters to increase the number of events that will be
% detected
pax0.eventtspan = 0.05;
pax0.maxamp = 4;
pax0.cmax = .02;
pax0.taft = 0.5;
pax0.tbef = 0.5;
% no display
pax0.display = 'none';
pax0.mlspikepar.dographsummary = false;

rng(seed,'twister')
spikes1 = spk_gentrain(rate,T,'bursty');
spikespad1 = [spk_gentrain(rate,T,'bursty') spikes1+T];

%% IV/1) Noise level
% The same factors that affect MLspike’s estimations affect also the
% autocalibration. Below we show that, e.g., different types of noises
% affect autocalibration differently: slow drifts (second row) have only
% little effect, as opposed to “worst band” (0.1-3Hz) noises, which affects
% autocalibration critically.. 
% For each estimation, purple arrows indicate the events detected by the
% autocalibration, their amplitude (?F/F in %) and the number of spikes
% they were assigned. Larger noise can lead to less events to be detected
% or to falsely detected ones, to approximate amplitude estimations and to
% wrong spike number assignments. Eventually this results in an erroneous
% estimation of A (see the indicated goodness of estimation expressed as
% ratios Aest/A and ?est/?), and in spike estimations errors.

%% IV/1.1) Photonic noise

figure(gcf);
fn_setfigsize(gcf,[1200,220])

ylim = [.65 1.65];

sigmas = [0.01 0.05];
plotnum = 120;
for i = 1 : length(sigmas)
    pcal1 = pcal; pcal1.sigma = sigmas(i);
    rng(seed,'twister');
    calciumpad1 = spk_calcium(spikespad1,pcal1); calcium1 = calciumpad1(nt+1:2*nt);
    % for display purpose:
    pax = pax0;
    pax.realspikes = {spikes1};
    pax.reala = pcal1.a;
    pax.realtau = pcal1.tau;
    [tauest aest sigmaest eventdesc] = spk_autocalibration({calcium1},pax);
    par1 = par;
    par1.tau = tauest;
    par1.a = aest;
    par1.finetune.sigma = sigmaest;
    par1.finetune.spikerate = rate;
    [spikest1 fit1 drift1] = spk_est(calcium1,par1);
    subplot(plotnum+i)
    spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
        'calciumeventsfull',eventdesc, ...
        'toptitle',sprintf('(RMS %g)',sigmas(i)));
    ht = findobj(gca,'type','text','units','normalized');
    set(ht,'string', ...
        [sprintf('A_{est}/A=%.2g, \\tau_{est}/\\tau=%.2g',aest/pcal.a,tauest/pcal.tau) ...
        ', ' get(ht,'string')])
    drawnow
end

%% IV/1.2) Low frequency noises (0.001Hz - 0.1 Hz)

figure(gcf);
fn_setfigsize(gcf,[1200,220])
ylim = [.65 1.65];

plotnum=120;
desired_rms = [0.05  0.2];
for i = 1 : 2
    rng(seed,'twister');
    pcal1 = pcal;
    calciumpad1 = spk_calcium(spikespad1,pcal1); calcium1 = calciumpad1(nt+1:2*nt);
    rng(seed,'twister');
    noise = simul_noiseequalizer(nt,dt,[.001 0.1],1);
    noise = noise * desired_rms(i)/rms(noise);
    calcium1 = calcium1 + noise;
    pax = pax0;
    % for display purpose:
    pax.realspikes = {spikes1};
    pax.reala = pcal.a;
    pax.realtau = pcal.tau;
    [tauest aest sigmaest eventdesc] = spk_autocalibration({calcium1},pax);
    par1 = par;
    if (~isempty(aest) && ~isempty(tauest) && ~isempty(sigmaest))
        par1.tau = tauest;
        par1.a = aest;
        par1.finetune.sigma = sigmaest;
        par1.finetune.spikerate = rate;
        [spikest1 fit1 drift1] = spk_est(calcium1,par1);
        subplot(plotnum+i)
        spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
            'calciumeventsfull',eventdesc, ...
            'toptitle',sprintf('(RMS %g)',desired_rms(i)));
        ht = findobj(gca,'type','text','units','normalized');
        set(ht,'string', ...
            [sprintf('A_{est}/A=%.2g, \\tau_{est}/\\tau=%.2g',aest/pcal.a,tauest/pcal.tau) ...
            ', ' get(ht,'string')])
    else
        spikest1 = [];
        subplot(plotnum+i);
        spk_display(dt,{spikest1},{calcium1},ylim,'factorbox');
        text('units','normalized','pos',[.05 .95],'string','autocalibration failed: no event found')
    end
    drawnow
end

%% IV/1.3)

figure(gcf)
fn_setfigsize(gcf,[1200,220])
ylim = [0.65 1.65];

rmss = [0.01 0.02];
plotnum = 120;
for i = 2% 1 : length(rmss)
    desired_rms = rmss(i);
    rng(seed,'twister');
    pcal1 = pcal;
    calciumpad1 = spk_calcium(spikespad1,pcal1);
    calcium1 = calciumpad1(nt+1:2*nt);
    rng(seed,'twister');
    badnoise = simul_noiseequalizer(nt,dt,[.1 3],1);
    badnoise = badnoise * desired_rms/rms(badnoise);
    calcium1 = calcium1+badnoise;
    pax = pax0;
    % for display purpose:
    pax.realspikes = {spikes1};
    pax.reala = pcal.a;
    pax.realtau = pcal.tau;
    [tauest aest sigmaest eventdesc] = spk_autocalibration({calcium1},pax);
    par1 = par;
    subplot(plotnum+i)
    if (~isempty(aest) && ~isempty(tauest) && ~isempty(sigmaest))
        par1.tau = tauest;
        par1.a = aest;
        par1.finetune.sigma = sigmaest;
        par1.finetune.spikerate = rate;
        [spikest1 fit1 drift1] = spk_est(calcium1,par1);
        spk_display(dt,{spikes1 spikest1},{calcium1 fit1 drift1},ylim,'factorbox', ...
            'calciumeventsfull',eventdesc, ...
            'toptitle',sprintf('(RMS %g)',desired_rms));
        ht = findobj(gca,'type','text','units','normalized');
        set(ht,'string', ...
            [sprintf('A_{est}/A=%.2g, \\tau_{est}/\\tau=%.2g',aest/pcal.a,tauest/pcal.tau) ...
            ', ' get(ht,'string')])
    else
        spikest1 = [];
        spk_display(dt,{spikest1},{calcium1},ylim,'factorbox');
        text('units','normalized','pos',[.05 .95],'string','autocalibration failed: no event found')
    end
    drawnow
end

%% IV/2) Data length
% For the same reason autocalibration becomes less accurate at higher
% spiking rates (see Fig. 1f), as less isolated events can be detected.

rate1 = 1;
T1 = 60;
nt1 = T1/dt;

% spikes
ylim = [.8 1.5];
T2 = [20 60]; % time
sigmas1 = 0.04;

% display
figure(gcf)
fn_setfigsize(gcf,[1200,250])

clf, p = get(axes,'pos');
w1 = p(3)*.9*(T2(1)/sum(T2)); w2 = p(3)*.9*(T2(2)/sum(T2)); h = .75;
clf, ha = [axes('pos',[p(1:2) w1 h]) axes('pos',[p(1)+p(3)-w2 p(2) w2 h])];

for seed1 = 20 %[2 3 10 20 23 27 30]
    %     figure(seed1), fn_setfigsize(seed1,[1200,250]), set(seed1,'color','w')
    %     clf, p = get(axes,'pos');
    %     w1 = p(3)*.9*(T2(1)/sum(T2)); w2 = p(3)*.9*(T2(2)/sum(T2)); h = .75;
    %     clf, ha = [axes('pos',[p(1:2) w1 h]) axes('pos',[p(1)+p(3)-w2 p(2) w2 h])];

    disp(['seed ' num2str(seed1)])
    rng(seed1,'twister')
    spikes1 = spk_gentrain(rate1,T1,'bursty');
    spikespad1 = [spk_gentrain(rate1,T,'bursty') spikes1+T1];
    
    j = 1;
    for i = 1 : length(T2)
        nt2 = T2(i)/dt;
        % calcium with no noise
        pcal1 = spk_calcium('par',dt,'T',2*T1,'sigma',sigmas1(j),'saturation',.1, ...
            'drift.parameter',[5 .015]);
        rng(seed1,'twister');
        calciumpad1 = spk_calcium(spikespad1,pcal1);
        spikes2 = spikes1(spikes1<=T2(i));
        calcium1 = calciumpad1(nt1+1:nt1+nt2);
        calcium1 = calcium1(1:nt2);
        % estimation
        par1 = tps_mlspikes('par',dt,'saturation',.1, ...
            'drift.parameter',.01,'algo.interpmode','linear', ...
            'display','none','dographsummary',false);
        
        pax = spk_autocalibration('par');
        pax.dt = dt;
        pax.saturation = 0.1;
        % for display purpose:
        pax.realspikes = {spikes2};
        pax.reala = pcal1.a;
        pax.realtau = pcal1.tau;
        pax.display = 'none';
        pax.mlspikepar.dographsummary = false;
        pax.amin = 0.05;
        pax.amax = 0.2;
        pax.eventtspan = 0.05;
        pax.maxamp = 4;
        pax.cmax = .02;
        pax.taft = 0.5;
        pax.tbef = 0.5;
        % the 3 parameter value below are equal to the default values, but
        % it is better to write them here, since the default values might
        % change in the close future
        pax.costfactor = [1 .5 0];
        [tauest aest sigmaest eventdesc] = spk_autocalibration({calcium1},pax);
        axes(ha(i));
        
        if (~isempty(aest) && ~isempty(tauest) && ~isempty(sigmaest))
            par1.tau = tauest;
            par1.a = aest;
            par1.finetune.sigma = sigmaest;
            par1.finetune.spikerate = rate1;
            [spikest1 fit1 drift1] = spk_est(calcium1,par1);
            spk_display(dt,{spikes2 spikest1},{calcium1 fit1 drift1},ylim, ...
                'factorbox','toptitle',sprintf('(RMS = %g, T = %is)',sigmas1,T2(i)), ...
                'calciumeventsfull',eventdesc)
            ht = findobj(gca,'type','text','units','normalized');
            str = [sprintf('A_{est}/A=%.2g, \\tau_{est}/\\tau=%.2g',aest/pcal.a,tauest/pcal.tau) ...
                ', ' get(ht,'string')];
            set(ht,'string',str)
        else
            spikest1 = [];
            XLim = [0 T2(i)];
            spk_display(dt,{spikes2},{calcium1},ylim,'factorbox','calciumeventsfull',eventdesc);
            str = 'autocalibration failed: no event found';
            text('units','normalized','pos',[.05 .95],'string',str)
        end
        %title(sprintf('(T=%g) (RMS=%g) (seed=%g)',T2(i), pcal1.sigma,seed1))
        
        fprintf('T=%is -> %s\n',T2(i),str)
        drawnow
    end
    drawnow
end






