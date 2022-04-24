clear;
clc;

addpath('spgl1-2.1');
optsL1 = spgSetParms('verbosity',0);

rand('seed', 2020);
randn('seed', 2002);

% Parameter settings
indexsize = [2, 4, 6, 8, 10]; % problem size
repeats = 20;
maxiter = inf;
freq = 5000; % the frequency of print the results
mu = 0.95;
gamma = 0.05;
tol = 1e-4;
delta1 =0.5;
delta2 = 0.1;
delta3 = 0.02;

lensize = length(indexsize);
table1 = [];

for h = 1 : lensize
    
    index = indexsize(h);
    m = 360 * index;
    n = 1280 * index;
    J = 2;
    k = 60 * index;     % sparsity of x0
    
    table2 = [];
    iter_sqp_s = 0; %Record the numbers of the iter of algorithm terminate by beta<1e-10
    iter_esqm_s1 = 0; %Record the numbers of the iter of algorithm terminate by t<1e-10
    iter_esqm_s2 = 0;
    iter_esqm_s3 = 0;
    
    for repeat = 1 : repeats
        [index, repeat]
        
        Are = randn(m, n); %A real part
        Aim = randn(m, n); %A imaginary part
        A = [Are -Aim; Aim Are];
        for j = 1 : 2*n
             A(:, j) = A(:, j)/norm(A(:, j));
         end
%        A = randn(m, n);
%        for j = 1 : n
%            A(:, j) = A(:, j)/norm(A(:, j));
%        end
        
        % generate the original signal
        I = randperm(n); 
        E = I(1 : k); 
        xre = zeros(n, 1);
        xre(E) = randn(k, 1); 
        xim = zeros(n, 1);
        xim(E) = randn(k, 1);
        x0 = [xre; xim];
%        I = randperm(n);
%        J = I(1 : k);
%        x0 = zeros(n, 1);
%        x0(J) = randn(k, 1);

        noisere = 0.005*tan(pi*(rand(m, 1) - 1/2));  %noise real part
        noiseim = 0.005*tan(pi*(rand(m, 1) - 1/2));  %noise imaginary part
        noise = [noisere; noiseim];
        b = A*x0 + noise;
        tmpvalue =  sum(log(1 + noise.^2/gamma^2));
        sigma = 1.2*tmpvalue;
        if sigma >= sum(log(1 + b.^2/gamma^2))
            error('0 is included in the feasible set. \n');
        end
        
        % Genarate the slater point
        tqr = tic;
        [Q,R] = qr(A',0);
        t_qr = toc(tqr);
        xslater = Q*(R'\b);
        t_slater = toc(tqr) - t_qr;
        fprintf(' time for QR %g, time for A\b generation %g\n', t_qr, t_slater)
        
        % Calculate the Lipschitz constant
        if m > 2000
            clear opts
            opts.issym = 1;
            tstart = tic;
            nmA = eigs(A*A', 1, 'LM');
            time_lambda = toc(tstart);
        else
            tstart = tic;
            nmA = norm(A*A');
            time_lambda = toc(tstart);
        end
        fprintf('\n Lipschitz constant L = %g\n', nmA)
        
        L = nmA;
        
        %  Using SPGL1 to genarate the start point
       fprintf('Start of SPGL1 \n');
        
        
        xstart =  feasible1(A, b, gamma, sigma, 0*xslater, xslater);
        [A1, b1, sigma1] = approximate(A, b, sigma, gamma, xstart); % Calculate the approximate model at xstart.
        tstart2 = tic;
        groups = zeros(2*n, 1) ; 
        for i = 1: 1: n
            groups(i, 1) = i;
            groups(i+n, 1) = i;
        end
        x_spgl1 = spg_group(A1, b1, groups, sigma1, optsL1);
        t_spgl1 = toc(tstart2);

        
        xmatrix_spgl1 = reshape(x_spgl1, n, J);
        fval_spgl1 = sum(sqrt(sum(xmatrix_spgl1'.*xmatrix_spgl1'))) - mu*norm(x_spgl1);
        Residual_spgl1 = (sum(log(1 + (A*x_spgl1 - b).^2/gamma^2)) - sigma)/sigma;        
        RecErr_spgl1 =  norm(x_spgl1 - x0)/max(1, norm(x0));
        fprintf(' SPGL1 terminated for l1 : time = %4.1f, nnz = %d,  fval = %7.4e, rec_err = %g, residual = %g \n',...
            t_spgl1, nnz(abs(x_spgl1) > 1e-10), fval_spgl1, RecErr_spgl1, Residual_spgl1);
        
        % Project x_spgl1 to the box C and check x_spgl1 belong to the feasible set
        xmatrix_spgl1 = 0.*xmatrix_spgl1;
        tfeas = tic;
        xslatermatrix = reshape(xslater, n, J);
        fval_xslater = sum(sqrt(sum(xslatermatrix'.*xslatermatrix'))) - mu*norm(xslater); 
        M =  fval_xslater/(1 - mu); % Upper bound of \|x_J\|
        normxmatrix = sqrt(sum(xmatrix_spgl1'.*xmatrix_spgl1'));
        JJ = normxmatrix > M;
        aa = ones(1, n);
        aa(JJ) = M./normxmatrix(JJ);
        xmatrix_spgl11 = xmatrix_spgl1'.*repmat(aa, J, 1);
        x_spgl11 = reshape(xmatrix_spgl11', 2*n, 1);   
        
        xin =  feasible1(A, b, gamma, sigma, x_spgl11, xslater);
        t_feas = toc(tfeas);
        
        % SQP
        tsqp = tic;
        [x_sqp, iter_sqp, flag_sqp] = L1L2_Lor_SQP_retract(A, b, gamma, sigma, mu, xin, xslater, L, M, maxiter, freq, tol, n, J);
        t_sqp = toc(tsqp);
        
        if flag_sqp ==1
            iter_sqp_s = iter_sqp_s +1;
        end
        xmatrix_sqp = reshape(x_sqp, n, J);
        fval_sqp = sum(sqrt(sum(xmatrix_sqp'.*xmatrix_sqp'))) - mu*norm(x_sqp);
        Residual_sqp = (sum(log(1 + (A*x_sqp - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_sqp = norm(x_sqp - x0)/max(1, norm(x0));
        fprintf('SQP Termination: iter = %d, time = %4.1f, nnz = %d,  fval = %16.10f, residual  = %7.4e, rec_err = %g \n',...
            iter_sqp, t_sqp, nnz(abs(x_sqp) > 1e-10), fval_sqp, Residual_sqp, RecErr_sqp)
        
        % ESQM with delta1
        tesqm1 = tic;
        [x_esqm1, iter_esqm1, flag_esqm1] = L1L2_Lor_ESQM_ls(A, b, gamma, sigma, mu, 0*xin, delta1, L, M, maxiter, freq, tol, n, J);
        t_esqm1 = toc(tesqm1);
        
        if flag_esqm1 == 1
            iter_esqm_s1 = iter_esqm_s1 +1;
        end
        xmatrix_esqm1 = reshape(x_esqm1, n, J);
        fval_esqm1 = sum(sqrt(sum(xmatrix_esqm1'.*xmatrix_esqm1'))) - mu*norm(x_esqm1);
        Residual_esqm1 = (sum(log(1 + (A*x_esqm1 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqm1 = norm(x_esqm1 - x0)/max(1, norm(x0));
        fprintf(' ESQM_1  terminated :  iter = %d, time = %4.1f, nnz = %d,  fval = %16.10f, residual = %7.4e, rec_err = %g  \n',...
            iter_esqm1, t_esqm1, nnz(abs(x_esqm1) > 1e-10), fval_esqm1, Residual_esqm1, RecErr_esqm1);
        
        % ESQM with delta2
        tesqm2 = tic;
        [x_esqm2, iter_esqm2, flag_esqm2] = L1L2_Lor_ESQM_ls(A, b, gamma, sigma, mu, 0*xin, delta2, L, M, maxiter, freq, tol, n, J);
        t_esqm2 = toc(tesqm2);
        
        if flag_esqm2 == 1
            iter_esqm_s2 = iter_esqm_s2 +1;
        end
        xmatrix_esqm2 = reshape(x_esqm2, n, J);
        fval_esqm2 = sum(sqrt(sum(xmatrix_esqm2'.*xmatrix_esqm2'))) - mu*norm(x_esqm2);
        Residual_esqm2 = (sum(log(1 + (A*x_esqm2 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqm2 = norm(x_esqm2 - x0)/max(1, norm(x0));
        fprintf(' ESQM_2  terminated :  iter = %d, time = %4.1f, nnz = %d,  fval = %16.10f, residual = %7.4e, rec_err = %g  \n',...
            iter_esqm2, t_esqm2, nnz(abs(x_esqm2) > 1e-10), fval_esqm2, Residual_esqm2, RecErr_esqm2);
        
        % ESQM with delta3
        tesqm3 = tic;
        [x_esqm3, iter_esqm3, flag_esqm3] = L1L2_Lor_ESQM_ls(A, b, gamma, sigma, mu, 0*xin, delta3, L, M, maxiter, freq, tol, n, J);
        t_esqm3 = toc(tesqm3);
        
        if flag_esqm3 == 1
            iter_esqm_s3 = iter_esqm_s3 +1;
        end
        xmatirx_esqm3 = reshape(x_esqm3, n, J);
        fval_esqm3 = sum(sqrt(sum(xmatirx_esqm3'.*xmatirx_esqm3'))) - mu*norm(x_esqm3);
        Residual_esqm3 = (sum(log(1 + (A*x_esqm3 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqm3 = norm(x_esqm3 - x0)/max(1, norm(x0));
        fprintf(' ESQM_3  terminated :  iter = %d, time = %4.1f, nnz = %d,  fval = %16.10f, residual = %7.4e, rec_err = %g  \n',...
            iter_esqm3, t_esqm3, nnz(abs(x_esqm3) > 1e-10), fval_esqm3, Residual_esqm3, RecErr_esqm3);
        
        % save the results
        
       table2 = [table2; t_qr, t_slater, t_spgl1, t_sqp, t_esqm1, t_esqm2, t_esqm3,  iter_sqp, iter_esqm1, iter_esqm2, iter_esqm3,...
            RecErr_spgl1, RecErr_sqp, RecErr_esqm1, RecErr_esqm2, RecErr_esqm3, Residual_spgl1, Residual_sqp, Residual_esqm1, Residual_esqm2, Residual_esqm3];
        
    end
    
table1 = [table1; mean(table2), iter_sqp_s, iter_esqm_s1, iter_esqm_s2, iter_esqm_s3];
    
end


% Save the results as columns
table1 = table1';

a = clock;
fname = ['Results\GL_LorNor_zeros_table' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
fid = fopen(fname, 'w');


for ii = 1:7
    fprintf(fid, '& %6.2f & %6.2f & %6.2f & %6.2f & %6.2f\n', table1(ii,:));
end
for ii = 8:11
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f\n', table1(ii,:));
end
for ii = 12:16
   fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f\n', table1(ii,:));
end
for ii = 17:21
    fprintf(fid, '& %3.2e & %3.2e & %3.2e & %3.2e & %3.2e\n', table1(ii,:)); 
end
for ii = 22:25
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f\n', table1(ii,:)); 
end
fclose(fid);