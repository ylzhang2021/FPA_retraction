function x = feasible1(A, b, gamma, sigma, x0, xslater)

% If x0 is not a feasible, retract x0 to boundary. 
% \| b - A*xslater\|_{LL2,gamma}  < sigma;
% \| b - A*x0\|_{LL2,gamma} > sigma;
% there exsits tt \in (0,1), such that x = x0 + tt*(xslater - x0) with \| b - A*x\|_{LL2,gamma} = sigma

x = x0;
Ax = A*x;
% Axslater = A*xslater;
tmp = Ax - b;
ellx = sum(log(1 + tmp.^2/gamma^2));
if ellx - sigma > 1e-12
    [~, s] = rootfind((Ax - b).^2, gamma^2, sigma);
    tt = 1 - s^1/2;
    x = x + tt*(xslater - x);
end

