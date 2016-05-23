function [ theta ] = leastabsdev(x,y)
N = size(x, 1);
M = size(x, 2);

f = [ zeros(M,1) ; ones(N,1) ];
A = [ -x -1*eye(N) ; x -1*eye(N) ];
b = [ -y ; y ];
sol = linprog(f,A,b);
theta = sol(1:M);
end