function [x, tstar] = rootfind(x0, epsilon, sigma)
% Finding the root of  \sum(log(1 + t*x0/epsilon)) = sigma (Note that x0>= 0)
% throught Newton's method.

t = 0;
fval = sigma - sum(log(1 + t*x0/epsilon));
grad = -sum(x0./(epsilon + t*x0));
while 1 == 1
    
    ttest = t - fval/grad;
    fvaltest = sigma - sum(log(1 + ttest*x0/epsilon));
    if abs(fvaltest)/sigma < 1e-10
        x = ttest*x0;
        tstar = ttest;
        break
    end
    t = ttest;
    fval = fvaltest;
    grad = -sum(x0./(epsilon + t*x0));
end
