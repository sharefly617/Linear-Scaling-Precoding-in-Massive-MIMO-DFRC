clc; clear; close all;
% rng(16);
%% ----------------- System Parameters -----------------
K     = 12;      % number of CUs
Mls     = 128;      % BS antennas
lm = length(Mls);
mcmc= 1;
Time = zeros(lm,mcmc);
for ind = 1:lm
    M = Mls(ind);
    for mc = 1:mcmc
D_k   = 2;      % streams per CU
N_k   = 2;      % antennas per CU
P_0   = 1;      % total TX power
sigmak = 1e-3;  % CU noise power
D     = D_k*K;
N     = N_k*K;

N_r = 16;       % sensing RX antennas

a_t = @(theta) exp(1j*pi*(0:M-1).'*sind(theta));
a_r = @(theta) exp(1j*pi*(0:N_r-1).'*sind(theta));

sigma_r = 1;    % sensing noise power
SNR = 10;
INR = 20;
a0 = sqrt(sigma_r*10^(SNR/10));
ai = sqrt(sigma_r*10^(INR/10));
tar     = 30;   % target angle (deg)
inter_vec = [-45, 10];   % multiple interference angles (deg)
J_int   = numel(inter_vec);  % number of interferences

%% ----------------- Sensing Channels -----------------
G0 = a0*a_r(tar) * a_t(tar)';      % target channel N_r x M
G_int = cell(J_int,1);
Rc = 0;
for j = 1:J_int
    G_int{j} = ai*a_r(inter_vec(j)) * a_t(inter_vec(j))';
    Rc = Rc+G_int{j};
end

%% ----------------- Communication Channels -----------------
H  = (randn(N,M)+1j*randn(N,M))/sqrt(2);    % N × M
H3d = zeros(N_k,  M,K);                     % Nk × M × K
for k=1:K
    rows = (k-1)*N_k+1:k*N_k;
    H3d(:,:,k) = H(rows,:);
end

%% Stack all channels for RWMMSE
% G    = [H; G0; vertcat(G_int{:})];           % (N + (1+J_int)*N_r) x M
% G    = [H; G0;Rc];
 G    = [H; a_t(tar)';a_t(inter_vec)'];
 % G    = H;
Ghat = G*G';
Ns_tot = size(G,1);

% F0, Fj = Gj * G^H
F0    = G0 * G';
F_int = cell(J_int,1);
for j = 1:J_int
    F_int{j} = G_int{j} * G';
end

%% ----------------- Rate Constraints & Dual Variables -----------------
SINR_dB  = 15;                       % SINR threshold per CU
SINR_min = 10^(SINR_dB/10);         % linear
Rmin     = log2(1 + SINR_min)*ones(K,1);   % R_k >= Rmin_k

lambda   = 0.01*ones(K,1);              % dual variables (per-user)
% lambda_max = 2*ones(K,1);
% lambda_min = 0*ones(K,1);
% stepsize0 = 1e-2;                   % base step size for dual ascent

%% ----------------- Initialize X -----------------
X = randn(Ns_tot,D) + 1j*randn(Ns_tot,D);
scale = sqrt(P_0/real(trace(Ghat*X*X')));
X = scale*X;

W      = repmat(eye(D_k),1,1,K);
W_s    = eye(D);
W_prev = W;

maxit = 500;
Mcell = cell(K,1);
Rrec  = zeros(maxit,1);
SCNR  = zeros(maxit,1);
SMI = zeros(maxit,1);
t1 = tic;
for it =1:maxit
    %% ====== 1) Update sensing receiver U_s and weight W_s ======
    Rs_cov = F0*X*X'*F0';
    for j = 1:J_int
        Rs_cov = Rs_cov + F_int{j}*X*X'*F_int{j}';
    end
    Rs_cov = Rs_cov + (sigma_r/P_0)*trace(Ghat*X*X')*eye(N_r);

    U_s = Rs_cov \ (F0*X);            % N_r×D
    W_s = inv(eye(D) - U_s'*F0*X);    % D×D

    %% ====== 2) Build X3d (each user's block) ======
    X3d = zeros(Ns_tot, D_k, K);
    for i=1:K
        cols = (i-1)*D_k+1:i*D_k;
        X3d(:,:,i) = X(:,cols);
    end

    %% ====== 3) Update comm. receivers U_k, weights W_k, and M_k ======
    U = zeros(N_k,D_k,K);
    for k=1:K
        Ghatk = H3d(:,:,k)*G';    % 1×Ns_tot
        Xk    = X3d(:,:,k);       % Ns_tot×1

        R_k = sigmak * eye(N_k);
        for j = 1:K
            Xj = X3d(:,:,j);
            R_k = R_k + Ghatk*Xj*Xj'*Ghatk';
        end

        U(:,:,k) = R_k \ (Ghatk*Xk);                  % 1×1
        W(:,:,k) = inv(eye(D_k) - U(:,:,k)'*Ghatk*Xk); % 1×1
        Mcell{k} = U(:,:,k) * W(:,:,k) * U(:,:,k)';    % 1×1
    end

    %% ====== 4) Update X using dual-weighted comm. terms ======
    Ms  = U_s*W_s*U_s';                             % N_r×N_r
    Xk0 = (sigma_r/P_0)*trace(Ms)*Ghat + F0'*Ms*F0; % sensing target term
    for j = 1:J_int
        Xk0 = Xk0 + F_int{j}'*Ms*F_int{j};          % sensing interferences
    end

    % Add communication-related terms weighted by lambda_k
    for j = 1:K
        Ghatj = H3d(:,:,j)*G';
        Xk0 = Xk0 ...
             + lambda(j)*(sigmak/P_0)*trace(Mcell{j})*Ghat ...
             + lambda(j)*Ghatj'*Mcell{j}*Ghatj;
    end

    temp = F0'*U_s*W_s;   % Ns_tot×D
    for k =1:K
        cols  = (k-1)*D_k+1:k*D_k;
        Ghatk = H3d(:,:,k)*G';
        X3d(:,:,k) = Xk0 \ ( lambda(k)*Ghatk'*U(:,:,k)*W(:,:,k) + temp(:,cols) );
        X(:,cols)  = X3d(:,:,k);
    end

    %% ====== 5) Power normalization and precoder P ======
    beta = P_0/real(trace(Ghat*(X*X')));
    P    = sqrt(beta)*G'*X;                      % M × D

    %% ====== 6) Compute SCNR (multi-interference) and user rates Rk ======
    % interference+noise covariance at sensing RX
    R_int = zeros(N_r);
    for j = 1:J_int
        R_int = R_int + G_int{j}*(P*P')*G_int{j}';
    end
    R_int = R_int + sigma_r*eye(N_r);

    SSINR    = real(trace( (G0*(P*P')*G0') / R_int ));
    SMI(it) = real(log2(det(eye(N_r)+(G0*(P*P')*G0') / R_int)));
    SCNR(it) = SSINR;

    Rk       = sumrate_correct(H,P,K,sigmak,N_k,D_k);
    Rrec(it) = sum(Rk);

    %% ====== 7) Dual ascent on lambda to enforce Rk >= Rmin ======
    viol     = Rmin - Rk;              
    % stepsize = stepsize0.*abs(viol);      % diminishing step size
    % lambda   = lambda + 0.1*viol;  % lambda_k >= 0
    % lambda   = max(lambda, 0);
    % for i = 1:K
    %     if viol(i)>0
    %         lambda(i)   = lambda(i) + (0.35/it)*viol(i);
    %     else
    %         lambda(i)   = lambda(i) + (0.1/it)*viol(i);
    %     end
    % end
    step = 0.001/ sqrt(it);         % 
    
    lambda = lambda + step * viol; % λ_{k+1} = λ_k + α * c_k   
    lambda   = max(lambda, 0);
    % lambda = (lambda_max-lambda_min)/2;
    % lambda = 0.001+zeros(K,1);
    %% ====== 8) Convergence check (objective + constraint violation) ======
    obj      = sum(arrayfun(@(kk) log10(real(det(W(:,:,kk)))), 1:K));
    obj_prev = sum(arrayfun(@(kk) log10(real(det(W_prev(:,:,kk)))),1:K));
    rel_obj  = abs(obj - obj_prev)/max(abs(obj_prev),1e-6);

    max_viol = max(Rmin - Rk);   % should be <= 0 ideally

    % fprintf(['Iter %3d: step = %.3e, sum-rate = %.4f, SCNR = %.4f, ', ...
    %          'min(Rk) = %.4f, max(Rmin-Rk) = %.4e, lambda = [%s]\n'], ...
    %         it, stepsize, Rrec(it), SCNR(it), min(Rk), max_viol, sprintf('%.3f ', lambda));

    % if it > 300 && rel_obj <= 1e-6 &&max_viol<=0
    %     % fprintf('Converged at iter %d with constraints nearly satisfied.\n', it);
    %     break;
    % end
    if it > 100 && abs(SCNR(it)-SCNR(it-1))/SCNR(it) <= 1e-5 &&max_viol<=1e-3
    %     % fprintf('Converged at iter %d with constraints nearly satisfied.\n', it);
        break;
    end

    W_prev = W;
end
Time(ind,mc)=toc(t1);
    end
end
time = mean(Time,2);
% semilogy(Mls,time,'b-x',linewidth=1.5);
it_end = it-1;




SCNR = pow2db(SCNR);

% %% ========== Plot convergence ==========
f2 = figure(2);
colororder({'k','b'})
yyaxis left
plot(1:it_end,Rrec(1:it_end), '-k', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('Sum Rate (bps/Hz)');

grid on;
hold  on
yyaxis right
plot(1:it_end,SMI(1:it_end), '-b', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('Iterations');
ylabel('Sensing MI (bps/Hz)');
grid off;


% 
% 
% 
% 
% 
% 
% %% ========== Beampattern ==========
% 
theta_scan = -90:0.5:90;              % angle grid
BP = zeros(size(theta_scan));

Rxx = P*P';                            % transmit covariance M×M

for idx = 1:length(theta_scan)
    at = a_t(theta_scan(idx));        % M×1
    BP(idx) = real(at' * Rxx * at);   % spatial gain
end

BP = BP / max(BP);
BP_dB = 10*log10(BP + 1e-12);

f3 = figure;
plot(theta_scan, BP_dB, 'LineWidth', 1.5); hold on;
xline(tar, '--r', 'LineWidth', 1.2);
for j = 1:J_int
    xline(inter_vec(j), '--g', 'LineWidth', 1.2);
end
hold off;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title(sprintf('SCNR = %.3f dB\n', pow2db(SSINR)));
legend_entries = [{'Beampattern','Target'}, ...
                  arrayfun(@(x) sprintf('Inter %d',x),1:J_int,'UniformOutput',false)];
legend(legend_entries{:});
grid on;



function R = sumrate_correct(H,P,K,sigma2,N_k,D_k)
[~,M] = size(H);
% D     = size(P,2);

P3d = zeros(M,D_k,K);
for k=1:K
    cols = (k-1)*D_k+1:k*D_k;
    P3d(:,:,k) = P(:,cols);                  % M × D_k
end

H3d = zeros(N_k,  M,K); 
R = zeros(K,1);
for k=1:K
    rows = (k-1)*N_k+1:k*N_k;
    H3d(:,:,k) = H(rows,:);
    Hk = H3d(:,:,k);
    Pk = P3d(:,:,k);

    Ck = zeros(N_k,N_k);
    for j=1:K
        Pj = P3d(:,:,j);
        Ck = Ck + Hk*(Pj*Pj')*Hk';
    end
    Sk = Hk*(Pk*Pk')*Hk';
    Ik = Ck - Sk;

    R(k) = real(log2(det( eye(N_k) + Sk/(Ik + sigma2*eye(N_k)) )));
end
end