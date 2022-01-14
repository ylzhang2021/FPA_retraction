function [xstar, lambda] = subprob_L1L2_ESQM(mu, a, sigma, alpha, x0, M, lambda)
% This aims to find the minimizer of the following problem
% min \|x\|_1 - mu*<xi, x - x0> + alpha*t + alpha/2 *\|x - x0\|^2
% s.t. <a, x> - sigma <= t && \|x\|_inf <= M && t >= 0

% Input
%
% mu           - real number in  [0,1)
% a              - n by 1 vector measurement
% sigma      - real number > 0
% alpha       - real number > 0
% x0            - n by 1 vector measurement
% M            - real number > 0
% lambda   - real number which is the starting point of lambda
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

gamma = 1/alpha;

%Calculate x^* when lambda = 0
tmp0 = x0 + gamma*xi;
xstar0 = sign(tmp0).*min(max(abs(tmp0) - gamma, 0), M);
g0 = sigma - a'*xstar0;

%Calculate x^* when lambda = alpha
tmpalpha = x0 + gamma*xi - a;
xstaralpha = sign(tmpalpha).*min(max(abs(tmpalpha) - gamma, 0), M);
gbeta = sigma - a'*xstaralpha;


if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = xstar0;
    return
elseif gbeta < tol % Check lambda = alpha?
    lambda = alpha;
    xstar = xstaralpha;
    return
else
    [xstar, lambda] = Newton_Monotone_LL(xi, a, sigma, gamma, x0, M, lambda, tol);
end

end