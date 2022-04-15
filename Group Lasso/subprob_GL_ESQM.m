function [xstar, lambda] = subprob_GL_ESQM(x0, a, sigma, mu, alpha, lambda, J, M)

% This aims to find the minimizer of the following problem
% min  \sum_{J\in\mathcal{J}} \|x_J\| - mu*<xi, x - x0> + alpha*t + alpha/2\|x - x0\|^2
% s.t. <a, x> - sigma <= t && \|x\|_inf <= M && t >= 0

% Input
%
% x0            - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma      - real number > 0
% mu         - real number  [ 0.95 ]
% alpha     - real number > 0
% lambda   - real number which is the initial of lambda
% J              - a positive integer whic denote the size of each block
% M           - real number > 0
%
%
% Output
%
% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

tol = 1e-10;
n = size(x0,1);

if norm(x0) <= tol
    xi = 0*x0;
else
    xi = mu*x0/norm(x0);
end

gamma = 1/alpha;
xmatrix = reshape(x0 + gamma*xi, J, n/J);
amatrix = reshape(a, J, n/J);

%Calculate x^* when lambda = 0?
tmp0 = xmatrix;
normtmp0 = sqrt(sum(tmp0.*tmp0));
xstarmatrix0 = repmat(min(max(1 - gamma./normtmp0, 0), M./normtmp0), J, 1).*tmp0;
g0 = sigma - sum(sum(amatrix.*xstarmatrix0));

%Calculate x^* when  lambda = alpha?
tmp1 = xmatrix - amatrix;
normtmp1 = sqrt(sum(tmp1.*tmp1));
xstarmatrix1 = repmat(min(max(1 - gamma./normtmp1, 0), M./normtmp1), J, 1).*tmp1;
g1 = sigma - sum(sum(amatrix.*xstarmatrix1));

if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = reshape(xstarmatrix0, n, 1);
    return
elseif g1 < tol % Check  lambda = alpha?
    lambda = alpha;
    xstar = reshape(xstarmatrix1, n, 1);
    return
else
    [xstar, lambda] = Newton_Monotone_GL(x0, a, sigma, xi, gamma, lambda, J, M, tol);
end

end