function [spk fit drift parest] = spk_est(calcium,par)
% function [spk fit drift parest] = spk_est(calcium,par)
% function par = spk_est('par')

if ischar(calcium) && strcmp(calcium,'par')
    par = tps_mlspikes('par');
    spk = par;
else
    defaultpar = tps_mlspikes('par');
    if isnumeric(par)
        dt = par;
        par = defaultpar;
        par.dt = par;
    else
        par = fn_structmerge(defaultpar,par);
    end
    [n fit parest dum dum drift] = tps_mlspikes(calcium,par); %#ok<*ASGLU>
    switch lower(par.algo.estimate)
        case 'map'
            spk = fn_timevector(n,par.dt);
        case 'proba'
            spk = n;
        case {'sample' 'samples'}
            spk = fn_timevector(n,par.dt);
    end
end