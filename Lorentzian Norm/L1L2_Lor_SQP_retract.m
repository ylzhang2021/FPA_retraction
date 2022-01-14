function [x, iter, flag] = L1L2_Lor_SQP_retract(A, b, gamma, sigma, mu, xstart, xslater, L, M, maxiter, freq, tol)

% This aims to use SQP_retract to find the minimizer of the following problem (involve Lorentzion norm)
% min \|x\|_1 - mu \|x\|_2
% s.t. \|Ax - b\|_{LL_2,gamma} <= sigma &&  \|x\|_inf <= M;
% Using SQP with retract.

% Input
%
% A            - m by n matrix (m << n)
% b             - m by 1 vector measurement
% gamma   - real number > 0
% sigma      - real number > 0
% mu          - real number in  [0,1)
% xstart      - starting point
% xslater     - a point verifying A*xslater = b
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
c = 1e-4;
beta_init = 1;
lambda = 0;
iter = 0;
flag = 0;

% Compute function value and gradient at start point
x = xstart;
Ax = A*x;
tmp = Ax - b;
ellx = sum(log(1 + tmp.^2/gamma^2)); % value of \ell at x
wx = 1./(gamma^2 + tmp.^2);
newsigma = sigma - ellx + wx'*tmp.^2; % value of \tilde{\sigma}
grad = 2*(A'*(tmp.*wx)); % gradient of \ell
newsigma1 = sigma - ellx + grad'*x; % value of subproblem's inequality constraint
fval = norm(x,1) - mu*norm(x);
Axslater = A*xslater;

fprintf(' ****************** Start  SQP_retract ********************\n')
fprintf('  iter   iter1      fval              fvalxtest                 err1                err2          norm(u - x) \n')

while 1 == 1
    beta = beta_init;
    iter1 = 0;
    
    while 1 == 1
        [u, lambda] = subprob_L1L2_SQP(mu, grad, newsigma1, beta, x, M, lambda); % Solving the subproblem
        Au = A*u;
        utmp = Au - b;
        ellxu = wx'*utmp.^2;
        gvalu = ellxu - newsigma;
               
        %    retract to the convex approximate set
        if gvalu > 1e-14
            tao = 1 - sqrt(newsigma/ellxu);  
            xtest = u + tao*(xslater - u);
        else
            xtest = u;
            tao = 0;
        end
        
        fvalxtest = norm(xtest,1) - mu*norm(xtest);
        
        %  Armijo line search
        if fvalxtest - fval > - c/2*norm(u - x)^2 && beta > 1e-10
            beta = beta/2;
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
    err1 = mu*norm(xiu - xix) + (2*lambda*L/gamma^2 + 1/beta)*norm(u - x);
    err2 = max(abs(lambda*(ellu - sigma)), ellu - sigma) ;
    if mod(iter,freq) == 0
        fprintf(' %5d|%5d| %16.10f|%16.10f|%3.6e|%3.6e|%3.6e\n',iter, iter1, fval, fvalxtest, err1, err2, norm(u - x))
    end
    
    if max(err1, err2*100) <= tol*max(1, norm(u)) || beta <= 1e-10 || iter >= maxiter
        if beta <=1e-10
            flag = 1;
            fprintf(' Terminate due to small beta\n')
        end
        break
    end
    
    % Update iterations, gradient and function value
    x = xtest;
    Ax = Au + tao*(Axslater - Au);
    tmp = Ax - b;
    ellx = sum(log(1 + tmp.^2/gamma^2));
    wx = 1./(gamma^2 + tmp.^2);
    grad = 2*(A'*(tmp.*wx));
    newsigma = sigma - ellx + wx'*tmp.^2;
    newsigma1 = sigma - ellx + grad'*x;
    fval = fvalxtest;
    
    if iter1 >= 1
        beta_init = min(max(beta, 1e-8), 1e8);
    else
        beta_init = min(max(beta*2, 1e-8), 1e8);
    end
    
    iter = iter + 1;
    
end







