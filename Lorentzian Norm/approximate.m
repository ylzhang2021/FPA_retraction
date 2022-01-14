function [A1, b1, sigma1] = approximate(A, b, sigma, gamma, x0)

% approximate the model at x0.
x= x0;
Ax = A*x;
tmp = Ax - b;
wx = 1./(gamma^2 + tmp.^2);
ellx = sum(log(1 + tmp.^2/gamma^2));
sigma1 = (sigma - ellx + wx'*tmp.^2)^(1/2);
A1 = repmat(wx.^(1/2), 1, size(A, 2)).*A;
b1 = wx.^(1/2).*b;
end
