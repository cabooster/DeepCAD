function spikes = spk_gentrain(rate,T,varargin)
% function spikes = spk_gentrain(rate,T[,mode[,parameters...]][,'repeat',n])
%---
% Generate a simulated spike train
%
% Input:
% - rate        desired average spike rate in Hz
% - T           length of the train in secondes
% - mode        'fix-rate' (regular Poisson train) [=default]
%               or 'vary-rate' (Poisson train with non-constant rate)
%               or 'bursty' (regular occurences of bursts)
% - parameters  for 'vary-rate' mode: 2 scalars:
%               - time constant for the smoothing of the varying rate in
%                 second [default 5 seconds]
%               - ratio between 0 and 1 of the time with non-zero rate
%                 [default 0.7]
%               for 'bursty' mode: 2 scalars:
%               - average number of spikes per burst (a Poisson
%                 distribution will be used with parameter this number)
%                 [default 1]
%               - average inter-spike interval within the burst (this
%                 average value will be randomly spread using a Poisson
%                 distribution with parameter 10) [default 0.01s]
%               for 'periodic' mode: 1 scalar:
%               - precision: value between 0 (exactly periodic) and 1
%               (Poisson) [default .2]
% - 'repeat',n  repeat until n trains are genetated
%
% Output:
% - spikes      vector - times of spikes, or cell array of vectors when
%               repeat is requested
 
% Input
mode = 'fix-rate';
parameters = [];
nrepeat = 0;
k = 0;
while k<length(varargin)
    k = k+1;
    a = varargin{k};
    if ischar(a)
        switch a
            case {'fix-rate' 'vary-rate' 'bursty' 'periodic'}
                mode = a;
            case 'repeat'
                k = k+1;
                nrepeat = varargin{k};
            otherwise
                error argument
        end
    else
        parameters = [parameters a]; %#ok<AGROW>
    end
end
dorepeat = logical(nrepeat);
if ~dorepeat, nrepeat = 1; end

% Generate
if isscalar(nrepeat), spikes = cell(1,nrepeat); else spikes = cell(nrepeat); end
nrepeat = prod(nrepeat);
switch mode
    case 'periodic'
        if isempty(parameters), precision=.2; else precision = parameters(1); end
        period = 1/rate;
        if precision<0 || precision>1, error 'wrong precision parameter', end
        for k=1:nrepeat
            % poissonian isi
            isi = exprnd(period);
            while sum(isi)<T, isi = [isi exprnd(period,1,round(rate*T))]; end
            % make them more "regular"
            isi0 = [period*rand period*ones(1,length(isi)-1)];
            isi = (1-precision)*isi0 + precision*isi;
            % get spike times
            spkk = cumsum(isi);
            spikes{k} = spkk(spkk<T);
        end
    case 'fix-rate'
        % easy one! first tell how many spikes
        nspike = poissrnd(rate*T,1,nrepeat);
        % then get the spike times
        for k=1:nrepeat
            spikes{k} = sort(rand(1,nspike(k))*T);
        end
    case 'bursty'
        % parameters
        if length(parameters)>=1, nperburst = parameters(1); else nperburst = 1; end
        if length(parameters)>=2, isi = parameters(2); else isi = .01; end
        
        % generate 'burst' events, then generate spikes within these bursts
        burstrate = rate/nperburst;
        for k=1:nrepeat
            nburst = poissrnd(burstrate*T);
            bursts = cell(1,nburst);
            for i=1:nburst
                nspike = poissrnd(nperburst);
                if nspike>0
                    bursts{i} = rand*T + cumsum(isi*(poissrnd(10,1,nspike)/10));
                end
            end
            spikes{k} = sort([bursts{:}]);
        end
    case 'vary-rate'
        % parameters
        if length(parameters)>=1, smoothtime = parameters(1); else smoothtime = 5; end
        if length(parameters)>=2, rnonzero = parameters(2); else rnonzero = .7; end
        if rnonzero==0, error 'error, rate cannot always be zero!!', end
        if rnonzero==1, error 'error, rate cannot always be non-zero, use fix-rate instead', end
        
        % 1) generate the varying rate vector
        nsub = 20; % the vector will be sampled at nsub * smooth frequency
        dt = smoothtime/nsub;
        nt = ceil(T/dt);
        x = randn(nt+2*nsub,nrepeat);
        vrate = fn_filt(x,nsub);
        vrate = vrate(nsub+(1:nt),:);
        % calculate the std of every element in vrate
        % note that fn_filt(.,nsub) achieves a low-pass with cut-off
        % frequency 1/nsub by smoothing with a Gaussian kernel Gs of std 
        % s=nsub*HWHH/(2*pi), where HWHH is the half-width at half-height
        % of a Gaussian of std 1
        % In other word, one can consider that:
        % vrate(t) = Gs * x(t) = sum Gs(i) x(t-i)
        % Var(vrate(t)) = sum Gs(i)^2 = 1 / (2*sqrt(pi)*s)
        HWHH = sqrt(2*log(2));
        s = nsub*HWHH/(2*pi);
        sr = sqrt(1/(2*sqrt(pi)*s));
        % rescale to have a std of 1
        vrate = vrate / sr;
        % translate the rate vector by the value above which we are as
        % often as specified by the 'rnonzero' parameter
        thr = norminv(1-rnonzero);
        vrate = max(0,vrate-thr);
        % calculate the average of elements in the new vrate
        dx = 1e-3;
        x = thr:dx:5; % i don't see another way to do than numerical integration
        y = (x-thr) .* normpdf(x);
        avgr = sum(y)*dx;
        % rescale the rate vector to achieve requested average rate
        vrate = vrate * (rate/avgr);

        % 2) generate spikes
        % choose a sampling rate at which there will be at most 1 spike per
        % bin
        dt1 = 1/max(vrate(:))/100;
        nt1 = floor(T/dt1);
        vrate = interp1((0:nt-1)*dt,vrate,(0:nt1-1)'*dt1);
        % generate spikes
        nspike = (rand(nt1,nrepeat)<vrate*dt1);
        % get spike times, add a small jitter within each time bin
        for k = 1:nrepeat
            spikesk = row(find(nspike(:,k))-1)*dt1;
            spikes{k} = spikesk + dt1 * rand(size(spikesk));
        end
end
if ~dorepeat, spikes = spikes{1}; end
