x = -8:0.1:8;
clear y;

delta = 1.345;
y(1,:) = robust_func(x, 'huber', 'rho', delta);
y(2,:) = robust_func(x, 'huber', 'psi', delta);
y(3,:) = robust_func(x, 'huber', 'wgt', delta);
figure;
subplot(2,3,1);
plot(x,y)
title('huber')

delta = 4.685;
% delta = 1.548;
y(1,:) = robust_func(x, 'bisquare', 'rho', delta);
y(2,:) = robust_func(x, 'bisquare', 'psi', delta);
y(3,:) = robust_func(x, 'bisquare', 'wgt', delta);
subplot(2,3,2);
plot(x,y)
title('bisquare')

delta = 2.985;
y(1,:) = robust_func(x, 'cauchy', 'rho', delta);
y(2,:) = robust_func(x, 'cauchy', 'psi', delta);
y(3,:) = robust_func(x, 'cauchy', 'wgt', delta);
subplot(2,3,3);
plot(x,y)
title('cauchy')

delta = 2.11;
y(1,:) = robust_func(x, 'welsch', 'rho', delta);
y(2,:) = robust_func(x, 'welsch', 'psi', delta);
y(3,:) = robust_func(x, 'welsch', 'wgt', delta);
subplot(2,3,4);
plot(x,y)
title('welsch')

% a = 1.387;
% b = 1.5;
% c = 1.063;
a = 1;
b = 2;
c = 0;
y(1,:) = robust_func(x, 'ggw', 'rho', a, b, c);
y(2,:) = robust_func(x, 'ggw', 'psi', a, b, c);
y(3,:) = robust_func(x, 'ggw', 'wgt', a, b, c);
subplot(2,3,5);
plot(x,y)
title('ggw')

