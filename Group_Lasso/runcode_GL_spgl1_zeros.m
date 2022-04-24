% runcode
clear;
clc;
addpath('spgl1-2.1');
optsL1 = spgSetParms('verbosity',0);

randn('seed', 2020);
rand('seed', 2002);

% Parameter settings
indexsize = [2, 4,  6, 8, 10];% problem size
repeats = 20;
maxiter = inf;
freq = 10000; % the frequency of print the results
tol = 1e-4;
delta1 = 0.5;
delta2 = 0.1;
delta3 = 0.02;
mu = 0.95;

lensize = length(indexsize);

table1 = [];

for h = 1 : lensize
    
    index = indexsize(h);
    m = 720*index;
    n = 2560*index;
    J = 2;  % size of each block
    k = 120*index;% nonzero blocks
    
   table2 = [];
   iter_sqp_s = 0;
   iter_esqm_s1 = 0;
   iter_esqm_s2 = 0;
   iter_esqm_s3 = 0;
   
    for repeat = 1 : repeats
        [index, repeat]
        
        A = randn(m, n);
        for i = 1 : n
            A(:, i) = A(:, i)/norm(A(:, i));
        end
        
        % generate the original date
        I = randperm(n/J);
        I = I(k+1 : end);
        x0 = randn(J, n/J);
        x0(:, I) = 0;
        x0 = reshape(x0, n, 1);
        
        noise = 0.005*randn(m, 1);  % Gaussian noise
        b = A*x0 + noise;
        
        sigma = 1.2*norm(noise);
        
        % Generate the slater point
        tqr = tic;
        [Q,R] = qr(A',0);
        t_qr = toc(tqr);
        xslater = Q*(R'\b);
        t_slater = toc(tqr) - t_qr;
        fprintf(' time for QR %g, time for feasible point generation %g\n', t_qr, t_slater)
        
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
                
        % Using SPGL1 to generate the start point
        fprintf('Start of SPGL1 \n');
        tspgl1 = tic;
        groups = zeros(1, n);
        for i =1: 1 : (n/J)
            groups(((i - 1)*J + 1) : i*J) = i;
        end
        x_spgl1 = spg_group(A, b, groups, sigma, optsL1);
        t_spgl1 = toc(tspgl1);
        
        xmatrix_spgl1 = reshape(x_spgl1, J, n/J);
        fval_spgl1 = sum(sqrt(sum(xmatrix_spgl1.*xmatrix_spgl1))) - mu*norm(x_spgl1);
        Residual_spgl1 = (norm(A*x_spgl1-b) - sigma)/sigma;
        RecErr_spgl1 =  norm(x_spgl1 - x0)/max(1, norm(x0));
        fprintf(' SPGL1 terminated for l1 : time = %4.1f, nnz = %d,  fval = %7.4e, rec_err = %g, residual = %g \n',...
            t_spgl1, nnz(abs(x_spgl1) > 1e-10), fval_spgl1, RecErr_spgl1, Residual_spgl1);
        
        % Project x_spgl1 to the box C
        tfeas = tic;
        xslatermatrix = reshape(xslater, J, n/J);
        fval_xslater = sum(sqrt(sum(xslatermatrix.*xslatermatrix))) - mu*norm(xslater); 
        M =  fval_xslater/(1 - mu); % Upper bound of \|x_J\|
        normxmatrix = sqrt(sum(xmatrix_spgl1.*xmatrix_spgl1));
        JJ = normxmatrix > M;
        aa = ones(1, n/J);
        aa(JJ) = M./normxmatrix(JJ);
        xmatrix_spgl11 = xmatrix_spgl1.*repmat(aa, J, 1);
        x_spgl11 = reshape(xmatrix_spgl11, n, 1); 
        
        % Check it belong to the feasible set. If not, pull it to boundary of the feasible set
        xtmp = A*x_spgl11 - b;
        gvalx = norm(xtmp)^2 - sigma^2;
        if gvalx > 1e-14
            tao = 1 - sigma/norm(xtmp);
            xin = x_spgl11 + tao*(xslater - x_spgl11);
        else
            xin = x_spgl11;
        end

        t_feas = toc(tfeas);
                
        % SQP_retract
        tsqp = tic;
        [x_sqp, iter_sqp, flag_sqp] = GL_SQP_retract(A, b, sigma, mu, J, xin, xslater, L, M, maxiter, freq, tol);
        t_sqp = toc(tsqp) ;
        
        if flag_sqp ==1
            iter_sqp_s = iter_sqp_s +1;
        end
        xmatrix_sqp = reshape(x_sqp, J, n/J);
        fval_sqp= sum(sqrt(sum(xmatrix_sqp.*xmatrix_sqp))) - mu*norm(x_sqp);
        Residual_sqp = (norm(A*x_sqp - b) - sigma)/sigma;
        RecErr_sqp = norm(x_sqp - x0)/max(1, norm(x0));
        nnz_sqp = nnz(abs(x_sqp) > 1e-10);
        fprintf('SQP_retract Termination: iter =%d, time =%g, nnz = %g, fval =%16.10f, feas vio = %g, diff = %g,\n',...
            iter_sqp, t_sqp, nnz_sqp, fval_sqp, Residual_sqp, RecErr_sqp)
        
        % ESQM with delta1
        tesqm1 = tic;
        [x_esqm1, iter_esqm1, flag_esqm1] = GL_ESQM_ls(A, b, sigma, mu, J, 0*xin, delta1, L, M, maxiter, freq, tol);
        t_esqm1 = toc(tesqm1);
        
        if flag_esqm1 == 1
            iter_esqm_s1 = iter_esqm_s1 +1;
        end
        xmatrix_esqm1 = reshape(x_esqm1, J, n/J);
        fval_esqm1= sum(sqrt(sum(xmatrix_esqm1.*xmatrix_esqm1))) - mu*norm(x_esqm1);
        Residual_esqm1 = (norm(A*x_esqm1 - b) - sigma)/sigma;
        RecErr_esqm1 = norm(x_esqm1 - x0)/max(1, norm(x0));
        nnz_esqm1 = nnz(abs(x_esqm1) > 1e-10);
        fprintf(' ESQM_0.5  Termination: iter = %d, time = %g, nnz = %g, fval = %16.10f, feas vio = %g, diff = %g,\n',...
            iter_esqm1, t_esqm1, nnz_esqm1, fval_esqm1, Residual_esqm1, RecErr_esqm1)
        
        % ESQM with delta2
        tesqm2 = tic;
        [x_esqm2, iter_esqm2, flag_esqm2] = GL_ESQM_ls(A, b, sigma, mu, J, 0*xin, delta2, L, M, maxiter, freq, tol);
        t_esqm2 = toc(tesqm2);
        
        if flag_esqm2 == 1
            iter_esqm_s2 = iter_esqm_s2 +1;
        end
        xmatrix_esqm2 = reshape(x_esqm2, J, n/J);
        fval_esqm2= sum(sqrt(sum(xmatrix_esqm2.*xmatrix_esqm2))) - mu*norm(x_esqm2);
        Residual_esqm2 = (norm(A*x_esqm2 - b) - sigma)/sigma;
        RecErr_esqm2 = norm(x_esqm2 - x0)/max(1, norm(x0));
        nnz22 = nnz(abs(x_esqm2) > 1e-10);
        fprintf(' ESQM_0.1  Termination: iter = %d, time = %g, nnz = %g, fval = %16.10f, feas vio = %g, diff = %g,\n',...
            iter_esqm2, t_esqm2, nnz22, fval_esqm2, Residual_esqm2, RecErr_esqm2)
        
        % ESQM with delta3
        tesqm3 = tic;
        [x_esqm3, iter_esqm3, flag_esqm3] = GL_ESQM_ls(A, b, sigma, mu, J, 0*xin, delta3, L, M, maxiter, freq, tol);
        t_esqm3 = toc(tesqm3);
                
        if flag_esqm3 == 1
            iter_esqm_s3 = iter_esqm_s3 +1;
        end
        xmatrix_esqm3 = reshape(x_esqm3, J, n/J);
        fval_esqm3= sum(sqrt(sum(xmatrix_esqm3.*xmatrix_esqm3))) - mu*norm(x_esqm3);
        Residual_esqm3 = (norm(A*x_esqm3 - b) - sigma)/sigma;
        RecErr_esqm3 = norm(x_esqm3 - x0)/max(1, norm(x0));
        nnz_esqm3 = nnz(abs(x_esqm3) > 1e-10);
        fprintf(' ESQM_0.02 Termination: iter = %d, time = %g, nnz = %g, fval = %16.10f, feas vio = %g, diff = %g,\n',...
            iter_esqm3, t_esqm3, nnz_esqm3, fval_esqm3, Residual_esqm3, RecErr_esqm3)
        
        table2 = [table2; t_qr, t_slater, t_spgl1, t_sqp, 0, t_esqm1, t_esqm2, t_esqm3,  iter_sqp, iter_esqm1, iter_esqm2, iter_esqm3,...
            RecErr_spgl1, RecErr_sqp, RecErr_esqm1, RecErr_esqm2, RecErr_esqm3, Residual_spgl1, Residual_sqp, Residual_esqm1, Residual_esqm2, Residual_esqm3];
        
    end
    
    table1 = [table1; mean(table2), iter_sqp_s, iter_esqm_s1, iter_esqm_s2, iter_esqm_s3];
    
end

% Save the results as columns
table1 = table1';

% Calculat slater+spgl1+FPA
table1(5, :) = table1(2, :) + table1(3, :) + table1 (4, :);

a = clock;
fname = ['Results\GL_table_spgl1_zeros_sum' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
fid = fopen(fname, 'w');

for ii = 1:8
    fprintf(fid, '& %6.2f & %6.2f & %6.2f & %6.2f & %6.2f\n', table1(ii,:));
end
for ii = 9:12
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f\n', table1(ii,:));
end
for ii = 13:17
   fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f\n', table1(ii,:));
end
for ii = 18:22
    fprintf(fid, '& %3.2e & %3.2e & %3.2e & %3.2e & %3.2e\n', table1(ii,:)); 
end
for ii = 23:26
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f\n', table1(ii,:)); 
end
fclose(fid);