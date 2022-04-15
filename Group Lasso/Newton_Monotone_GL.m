function [xstar, lambda] = Newton_Monotone_GL(x0, amatrix, sigma, xi, gamma, lambda, J, M, tol)
% This aims to find the root of T(lambda) = 0;
% where T(lambda) = sigma - a^T*x;
% and x_J = min(max(1 - gamma./\|tmp_J\|,0), M./\|tmp_J\|), J, 1).*tmp_J ;
% and tmp_J = x0_J + gamma*xi_J - lambda*gamma*a_J.

% Initialization
eta = 1e-4; % parameter for regular
c = 1e-4;

n = size(x0,1);
iter = 0;

% Compute function value
x1 = x0 + gamma*xi;
x1matrix = reshape(x1, J, n/J);
amatrix = reshape(amatrix, J, n/J);
tmp = x1matrix - (lambda*gamma)*amatrix;
normtmp = sqrt(sum(tmp.*tmp));
xstarmatrix = repmat(min(max(1 - gamma./normtmp,0), M./normtmp), J, 1).*tmp;
g = sigma - sum(sum(amatrix.*xstarmatrix));
normg = abs(g);


if normg <= tol % Check  lambda = the initial lambda?
    xstar = reshape(xstarmatrix, n, 1);
    return
else    
    while normg > tol

        muk = eta*normg^(1/2);
        
        %  calculate the generalized Jacobian
        I = normtmp > gamma & normtmp <= gamma +M;  %  gamma< \|y_J\| <= gamma + M
        tmpmat = amatrix(:,I);
        tmp1 = tmp(:,I);
        normtmp1 = normtmp(I);
        H1 = gamma^2*sum(sum((tmp1.*tmpmat)).^2./(normtmp1.^3)) + gamma*sum((1 - gamma./normtmp1).*sum(tmpmat.^2));
        
        K = normtmp > gamma + M;  % \|y_J\| > gamma + M
        tmpmat1 = amatrix(:,K);
        tmp11 = tmp(:,K);
        normtmp11 = normtmp(K);
        H2 = - M*gamma*sum(sum((tmp11.*tmpmat1)).^2./(normtmp11.^3)) + M*gamma*sum((1./normtmp11).*sum(tmpmat1.^2));
        
        H = H1 + H2;
        
        % Calculate the direction
        H = H + muk;
        dir = H\(-g);
        
        % Linesearch 
        s = 1;
        while 1==1            
            u = lambda + s*dir;
            tmp = x1matrix - (u*gamma)*amatrix;
            normtmp = sqrt(sum(tmp.*tmp));
            xstarmatrix = repmat(min(max(1 - gamma./normtmp,0), M./normtmp), J, 1).*tmp;
            g = sigma - sum(sum(amatrix.*xstarmatrix));
            if g*dir + c*muk*dir^2 <= 0 || s < 1e-10
                break
            else
                s = s/2;
            end
        end

        lambda = u;
        normg = abs(g);
        
        if s < 1e-10
            break
        end
        iter = iter + 1;
    end
end
xstar = reshape(xstarmatrix, n, 1);



   