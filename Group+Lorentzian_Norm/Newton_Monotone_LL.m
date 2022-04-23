function [xstar, lambda] = Newton_Monotone_LL(xi, a, sigma, gamma, x0, M, lambda, tol)
% This aims to find the root of T(lambda) = 0;
% where T(lambda) = sigma - a^T*x;
% and x = sign(tmp)*min(max(abs(tmp) - gamma, 0), M);
% and tmp = x0 + gamma*xi - lambda*gamma*a.

% Initialization
eta = 1e-4; % parameter for regular 
c = 1e-4; 
iter = 0;

% Compute function value
x1 = x0 + gamma*xi;
tmp = x1 - lambda*gamma*a;
xstar = sign(tmp).*min(max(abs(tmp) - gamma, 0), M);
g = sigma - a'*xstar;
normg = abs(g);

while normg > tol
    
    muk = eta*normg^(1/2);
    
    %  calculate the generalized Jacobian
    I = abs(tmp) > gamma & abs(tmp) <= gamma +M;  %  gamma< \|y_J\| <= gamma + M
    a0 = a(I);
    H = gamma*norm(a0)^2;
    
    % Calculate the direction
    H = H + muk;
    dir = H\(-g);
    
    % Linesearch
    s = 1;
    while 1==1
        u = lambda + s*dir;
        tmp = x1 - u*gamma*a;
        xstar = sign(tmp).*min(max(abs(tmp) - gamma, 0), M);
        g = sigma - a'*xstar;
        if g*dir + c*muk*dir^2 <= 0 || s < 1e-10   %10
            break
        else
            s = s/2; %2
        end
    end
    
    lambda = u;
    normg = abs(g);
    
    if s < 1e-10 %10
        break
    end
    
    iter = iter + 1;
end

end


