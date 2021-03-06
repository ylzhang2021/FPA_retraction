function [xstar, lambda] = subprob_L1L2_SQP(mu, a, sigma, gamma, x0, M, lambda, n, J)
% This aims to find the minimizer of the following problem
% min sum\|x_J\| - mu*<xi, x - x0> + 1/(2*gamma)\|x - x0\|^2
% s.t. <a, x> <= sigma && \|x\|_inf <= M   

% Input
%
% mu           - real number in  [0,1)
% a            - n by 1 vector measurement
% sigma        - real number > 0
% gamma        - real number > 0  beta
% x0           - n by 1 vector measurement
% M            - real number > 0
% lambda       - real number which is the starting point of lambda
% n, J         - group size parameters. n groups, each of size J
%

% Output
%
% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

tol = 1e-10;

if norm(x0) <= tol
    xi = 0*x0;
else
    xi = mu*x0/norm(x0);
end

% Calculate x^* when lambda = 0
tmp0 = reshape(x0 + gamma*xi, n, J)';
normtmp0 = sqrt(sum(tmp0.* tmp0));
xstar0 = repmat(min(max(1-gamma./normtmp0, 0),M./normtmp0), J, 1).*tmp0;
g0 = sigma - sum(sum(reshape(a, n, J)'.*xstar0));

if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = reshape(xstar0', n*J, 1);
    return
else
    [xstar, lambda] = Newton_Monotone_GL(x0, a, sigma, xi, gamma, lambda, J, M, tol);
end

end