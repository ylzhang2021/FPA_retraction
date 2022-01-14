function [x, iter, flag] = GL_ESQM_ls(A, b, sigma, mu, J, xstart, delta, L, M, maxiter, freq, tol)

% This aims to find the minimizer of the following problem (Gruop LASSO)
% min \sum_{J\in\mathcal{J}} \|x_J\| - mu*norm(x)
% s.t. \|Ax - b\|^2 <= sigma^2 &&  \|x\|_inf <= M 
% Using ESQM method with line search.

% Input
%
% A            - m by n matrix (m << n)
% b             - m by 1 vector measurement
% sigma      - real number > 0
% mu          - real number in (0, 1)
% J              - a positive integer whic denote the size of each block
% xstart       - the starting point
% delta       - real number > 0
% L             - the Lipschitz constant
% M           - Upper bound of \|x_J\| for any J
% maxiter   - maximum number of iterations [inf]
% freq         - The frequency of print the results
% tol           - tolerance [1e-4]
%
%
% Output
%
% x            - approximate stationary point
% iter        - number of iterations
% flag       - a number of 0 or 1

% Initialization
rho = 1e-4;
beta_init = 1;
lambda = 0; % parameter for subproblem
iter = 0;
flag = 0;
n = size(xstart, 1);

% Compute function value and gradient
x = xstart; % starting point
Ax = A*x;
tmp = Ax - b;
gval = norm(tmp)^2 - sigma^2;
grad = 2*(A'*tmp);
xmatrix = reshape(x, J, n/J);
fval = sum(sqrt(sum(xmatrix.*xmatrix))) - mu*norm(x);


fprintf(' ****************** Start   ESQM ********************\n')
fprintf('  iter   iter1      fval            err1          err2         gvalu         beta         lambda         norm(u - x)       t\n')

while 1 == 1
    
    beta = beta_init;
    
    [u, lambda] = subprob_GL_ESQM(x, grad, grad'*x - gval, mu, beta, lambda, J, M); % Solving the subproblem
    Au = A*u;
    utmp = Au - b;
    gvalu = norm(utmp)^2 - sigma^2;
    
    % Line search
    fvaltest = fval + beta*max(0, gval);
    iter1 = 0;
    t = 1;
    while 1== 1
        xtest1 = x + t*(u - x);
        Axtest1 = Ax + t*(Au - Ax);
        tmp1 = Axtest1 - b;
        gvalxtest1 = norm(tmp1)^2 - sigma^2;
        xtestmatrix1 = reshape(xtest1, J, n/J);
        fvalxtest1= sum(sqrt(sum(xtestmatrix1.*xtestmatrix1))) - mu*norm(xtest1);
        fvaltest11 = fvalxtest1 + beta*max(0, gvalxtest1);
        
        if fvaltest11 - fvaltest  > -beta*rho*t*norm(u - x)^2 && t>1e-10
            t = t/2;
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
    
    err1 = mu*norm(xiu - xix) + (2*lambda*L + beta)*norm(u - x); % norm(u-x); %
    err2 = max(abs(lambda*gvalu), gvalu);
    sss = gval + grad'*(u - x);
    
    if mod(iter, freq) == 0
        fprintf(' %5d|%5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, iter1, fval, err1, err2, gvalu, beta, lambda, norm(u - x), t )
    end
    
    if max(err1, err2*100) <= tol*max(1, norm(u)) || t<=1e-10 || iter >= maxiter 
        if t <=1e-10
            flag = 1;
            fprintf(' Terminate due to small gamma\n')
        end
        break
    end
    
        
    % Update iterations, gradient and function value
    
    x = xtest1;
    Ax = Axtest1;
    tmp = tmp1;
    gval = gvalxtest1;
    grad = 2*(A'*tmp);
    fval = fvalxtest1;
    
    if sss > 1e-10
        beta_init = beta + delta;
    else
        beta_init = beta;
    end
    
    iter = iter + 1;
    
end






