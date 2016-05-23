function o = robust_func(z, functype, outtype, delta, b, c)
% functype: 'huber', 'bisquare', 'cauchy', 'welsch', 'ggw'
% outtype: 'rho', 'psi', 'wgt'

switch lower(functype)

    case('l1')
        switch lower(outtype)
            case('rho')
                o = abs(z);
            case('psi')
                o = sign(z);
            case('wgt')
                o = 1 ./ max(abs(z),0.000001);
            otherwise
                warning('error outtype');
        end
    
    case('huber')
        k = delta;
        switch lower(outtype)
            case('rho')
                k_mask = abs(z)<k;
                o = zeros(size(z));
                o(k_mask) = 0.5*(z(k_mask).^2);
                o(~k_mask) = k * (abs(z(~k_mask))-k/2);
            case('psi')
                k_mask = abs(z)<k;
                o = zeros(size(z));
                o(k_mask) = z(k_mask);
                o(~k_mask) = k * sign(z(~k_mask));
            case('wgt')
                o = min(1, k ./ abs(z));
            otherwise
                warning('error outtype');
        end

    case('bisquare')
        k = delta;
        switch lower(outtype)
            case('rho')
                k_mask = abs(z)<k;
                o = zeros(size(z));
                o(k_mask) = 1/6*k*k*(1-(1-(z(k_mask)/k).^2).^3);
                o(~k_mask) = 1/6*k*k;
            case('psi')
                k_mask = abs(z)<k;
                o = zeros(size(z));
                o(k_mask) = z(k_mask) .* (1-(z(k_mask)/k).^2).^2;
            case('wgt')
                k_mask = abs(z)<k;
                o = zeros(size(z));
                o(k_mask) = (1-(z(k_mask)/k).^2).^2;
            otherwise
                warning('error outtype');
        end

    case('cauchy')
        k = delta;
        switch lower(outtype)
            case('rho')
                o = 0.5*log(1+(z/k).^2);
            case('psi')
                o = 1*z ./ (1+(z/k).^2);
            case('wgt')
                o = 1 ./ (1+(z/k).^2);
            otherwise
                warning('error outtype');
        end

    case('welsch')
        k = delta;
        switch lower(outtype)
            case('rho')
                o = 1/k/k * exp(-0.5*(z/k).^2);
            case('psi')
                o = z .* exp(-0.5*(z/k).^2);
            case('wgt')
                o = exp(-0.5*(z/k).^2);
            otherwise
                warning('error outtype');
        end

    case('ggw')
        a = delta;
        switch lower(outtype)
            case('rho')
                o = 1;
            case('psi')
                c_mask = abs(z)<=c;
                o = zeros(size(z));
                o(c_mask) = z(c_mask);
                o(~c_mask) = z(~c_mask).*exp(-0.5*((z(~c_mask)-c).^b));
            case('wgt')
                c_mask = abs(z)<=c;
                o = zeros(size(z));
                o(c_mask) = 1;
                o(~c_mask) = exp(-0.5*((z(~c_mask)-c).^b));
            otherwise
                warning('error outtype');
        end

   otherwise
            warning('error functype');
end
