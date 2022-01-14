function [x, iter, flag] = GL_SQP_retract(A, b, sigma, mu, J, xstart, xslater, L, M, maxiter, freq, tol)

% This aims to find the minimizer of the following problem (Group LASSO)
% min \sum_{J\in\mathcal{J}} \|x_J\| - mu*norm(x)
% s.t. \|Ax - b\|^2 <= sigma^2 && \|x\|_inf <= M
% Using SQP with retract.

% Input
%
% A            - m by n matrix (m << n)
% b             - m by 1 vector measurement
% sigma      - real number > 0
% mu          - real number in (0, 1)
% J              - a positive integer whic denote the size of each block
% xstart       - the starting point
% xslater      - a point verifying A*xslater = b
% L             - the Lipschitz constant
% M           - Upper bound of \|x_J\| for any J
% maxiter   - maximum number of iterations [inf]
% freq         - The frequency of print the results
% tol           - tolerance [1e-4]
%

% Output
%
% x            - approximate stationary point
% iter        - number of iterations
% flag       - a constant of 0 or 1



% Initialization
c = 1e-4; % parameter for Armijo line search
beta_init = 1; % parameter for subproblem
lambda = 0; % parameter for subproblem
iter = 0;
flag = 0;

% Compute function value and gradient
n = size(xstart, 1);
x = xstart; % starting point
Ax = A*x;
tmp = Ax - b;
grad = 2*(A'*tmp);
gval = norm(tmp)^2 - sigma^2;
xmatrix = reshape(x, J, n/J);
fval = sum(sqrt(sum(xmatrix.*xmatrix))) - mu*norm(x);
Axslater = A*xslater;

fprintf(' ****************** Start   SQP_retract ********************\n')
fprintf('  iter   iter1       fval         err1         err2         beta         gval         lambda\n')

while 1 == 1
    beta = beta_init;
    iter1 = 0;
    
    while 1 == 1
        [u, lambda] = subprob_GL_SQP(x, grad, grad'*x - gval, mu, beta, lambda, J, M); % Solving the subproblem
        Au = A*u;
        utmp = Au - b;
        gvalu = norm(utmp)^2 - sigma^2;
        
        % retract step
        if gvalu > 1e-14
            tao = 1 - sigma/norm(utmp);
            xtest = u + tao*(xslater - u);
        else
            xtest = u;
            tao = 0;
        end
        xtestmatrix = reshape(xtest, J, n/J);
        fval1= sum(sqrt(sum(xtestmatrix.*xtestmatrix))) - mu*norm(xtest); % calculate the objective at xtest
        
        %  Armijo line search
        if fval1 - fval > - c/2*norm(u - x)^2 && beta>1e-10
            beta = beta/2;
            iter1 = iter1 +1;
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
    
    err1 = mu*norm(xiu - xix) + (2*lambda*L + 1/beta)*norm(u - x);
    err2 = max(abs(lambda*gvalu), gvalu);
    if mod(iter, freq) == 0
        fprintf(' %5d|%5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, iter1, fval, err1, err2, beta, gval, lambda)
    end
    
    if max(err1, err2*100) <= tol*max(1, norm(u))  || beta <= 1e-10 || iter >= maxiter
        if beta <=1e-10
            flag = 1;
            fprintf(' Terminate due to small gamma\n')
        end
        break
    end
    
    % Update iterations, gradient and function value
    x = xtest;
    Ax = Au + tao*(Axslater - Au);
    tmp = Ax - b;
    gval = norm(tmp)^2 - sigma^2;
    fval = fval1;
    grad = 2*(A'*tmp);
    
    if iter1 >= 1
        beta_init = min(max(beta, 1e-8), 1e8);
    else
        beta_init = min(max(beta*2, 1e-8), 1e8);
    end
    
    iter = iter + 1;
end





