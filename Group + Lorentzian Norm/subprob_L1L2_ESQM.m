function [xstar, lambda] = subprob_L1L2_ESQM(mu, a, sigma, alpha, x0, M, lambda, n, J)
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
xmatrix = reshape(x0 + gamma*xi, n, J)';
amatrix = reshape(a, n, J)';

%Calculate x^* when lambda = 0
tmp0 = xmatrix;
normtmp0 = sqrt(sum(tmp0.*tmp0));
xstarmatrix0 = repmat(min(max(1-gamma./normtmp0, 0), M./normtmp0), J, 1).*tmp0;   
g0 = sigma - sum(sum(amatrix.*xstarmatrix0));

%Calculate x^* when lambda = alpha
tmpalpha = xmatrix - amatrix;
normtmp1 = sqrt(sum(tmpalpha.*tmpalpha));
xstaralpha = repmat(min(max(1-gamma./normtmp1, 0), M./normtmp1), J, 1).*tmpalpha ;
gbeta = sigma - sum(sum(amatrix.*xstaralpha));


if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = reshape(xstarmatrix0', J*n, 1);
    return
elseif gbeta < tol % Check lambda = alpha?
    lambda = alpha;
    xstar = reshape(xstaralpha', J*n, 1);
    return
else
    [xstar, lambda] = Newton_Monotone_GL(x0, a, sigma, xi, gamma, lambda, J, M, tol);
end

end