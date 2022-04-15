function [xstar, lambda] = subprob_GL_SQP(x0, a, sigma, mu, gamma, lambda, J, M)

% This aims to find the minimizer of the following problem
% min \sum_{J\in\mathcal{J}} \|x_J\|  - mu*<xi, x - x0>+ 1/(2*gamma)\|x - x0\|^2
% s.t. <a, x> <= sigma &&  \|x\|_inf <= M

% Input
%
% x0            - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma      - real number > 0
% mu         - real number  [ 0.95 ]
% gamma   - real number > 0
% lambda   - real number which is the initial of lambda
% J              - a positive integer whic denote the size of each block
% M           - real number > 0

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

%Calculate x^* when lambda = 0?
tmp0 = reshape(x0 + gamma*xi, J, n/J);
normtmp0 = sqrt(sum(tmp0.*tmp0));
xstarmatrix0 = repmat(min(max(1 - gamma./normtmp0,0), M./normtmp0), J, 1).*tmp0;
g0 = sigma - sum(sum(reshape(a, J, n/J).*xstarmatrix0));

if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = reshape(xstarmatrix0, n, 1);
    return
else
    [xstar, lambda] = Newton_Monotone_GL(x0, a, sigma, xi, gamma, lambda, J, M, tol);
end



%     fprintf(' Subprob termination: iter %d, normg %g, lambda %g\n', iter, norm(g), lambda)

% % check the approximate solution with  solution of CVX
% cvx_begin quiet
%   variable u(J,n/J)
%   minimize sum(norms(u,2,1)) - mu*<xi, x - x0> + 1/(2*gamma)*sum(sum((u - x0).*(u - x0)))
%   subject to
%       sum(sum(a.*u)) <= sigma;
%       u(:) >= -M;
%       u(:) <= M;
% cvx_end
%
% dd = norm(reshape(u, n,1) - xstar)
end