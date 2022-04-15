function [x, iter, flag] = L1L2_Lor_ESQM_ls(A, b, gamma, sigma, mu, xstart, delta, L, M, maxiter, freq, tol, n, J)

% This aims to use ESQM_ls to find the minimizer of the following problem (involve Lorentzion norm)
% min \|x\|_1 - mu \|x\|_2
% s.t. \|Ax - b\|_{LL_2,gamma} <= sigma &&  \|x\|_inf <= M;
% Using ESQM method with line search.

% Input
%
% A            - m by n matrix (m << n)
% b             - m by 1 vector measurement
% gamma   - real number > 0
% sigma      - real number > 0
% mu          - real number in  [0,1)
% xstart      - starting point
% delta       -  real number > 0
% L             - a constant [ \|A\|^2 ]
% M           - Upper bound of x
% maxiter   - maximum number of iterations 
% freq         - The frequency of print the results
% tol           - tolerance [1e-4]
%
%
% Output
%
% x          - approximate stationary point
% iter      - number of iterations
% flag      - a constant of 0 or 1




% Initialization
c = 1e-4;  %1e-4
beta_init = 1;
lambda = 0;
iter = 0;
flag = 0;

% Compute function value and gradient at start point
x = xstart;
Ax = A*x;
tmp = Ax - b;
ellx = sum(log(1 + tmp.^2/gamma^2));
wx = 1./(gamma^2 + tmp.^2);
grad = 2*(A'*(tmp.*wx)); % gradient of \ell
newsigma1 = sigma - ellx + grad'*x; % value of subproblem's inequlity constraint
x_reim = reshape(x, n, J);
fval = sum(sqrt(sum(x_reim'.*x_reim'))) - mu*norm(x);

fprintf(' ****************** Start EQSM ********************\n')
fprintf('  iter   iter1      fval             err1              err2         norm(u - x)            beta               norm(grad)           t\n')

while 1 == 1
    beta = beta_init;
    t = 1;
    
    [u, lambda] = subprob_L1L2_ESQM(mu, grad, newsigma1, beta, x, M, lambda, n, J); % Solving the subproblem
    Au = A*u;
    utmp = Au - b;
        
    %  Line search
    fvalxtest = fval + beta*max(0, ellx - sigma);
    iter1 = 0;
    while 1 == 1
        xtest1 = x + t*(u - x);
        Axtest1 = Ax + t*(Au - Ax);
        tmp1 = Axtest1 - b;
        xtest1_reim = reshape(xtest1, n, J);
        fval1 = sum(sqrt(sum(xtest1_reim'.*xtest1_reim'))) - mu*norm(xtest1);
        ellxtest1 = sum(log(1 + tmp1.^2/gamma^2));
        fvalxtest1 = fval1 + beta*max(0, ellxtest1 - sigma);

        if fvalxtest1 - fvalxtest > -beta*c*t*norm(u - x)^2 && t > 1e-10
            t = t/2;
            iter1 = iter1 + 1;
        else
            break
        end
    end
    
    % check for termination
    if norm(u) <= 1e-10
        xiu = 0*u;
    else
        xiu = u/norm(u);
    end
    if norm(x) <= 1e-10
        xix = 0*x;
    else
        xix = x/norm(x);
    end
    
    ellu = sum(log(1 + utmp.^2/gamma^2)); % the value of ell at u
    err1 = mu*norm(xiu - xix) + (2*lambda*L/gamma^2 + beta)*norm(u - x);
    err2 = max(abs(lambda*(ellu - sigma)), ellu - sigma);
    sss = ellx + grad'*(u - x) - sigma;
    
    if mod(iter,freq) == 0
        fprintf(' %5d|%5d| %16.10f|%3.4e|%3.4e|%3.4e|%3.4e|%3.4e|%3.4e\n',iter, iter1, fval, err1, err2, norm( u - x), beta,norm(grad), t )
    end
    
    if max(err1, abs(err2)*100) <= tol*max(1, norm(u)) || t <= 1e-10 || iter >= maxiter 
        if t <=1e-10 %10
            flag = 1;
            fprintf(' Terminate due to small t\n')
        end
        break
    end
    
    % Update iterations, gradient and function value
        
    x = xtest1;
    Ax = Axtest1;
    tmp = tmp1;
    ellx = ellxtest1;
    wx = 1./(gamma^2 + tmp.^2);
    grad = 2*(A'*(tmp.*wx));
    newsigma1 = sigma - ellx + grad'*x;
    fval = fval1;
    
    if sss > 1e-10
        beta_init = beta + delta; 
    else
        beta_init = beta;
    end
    
    iter = iter + 1;    
end







