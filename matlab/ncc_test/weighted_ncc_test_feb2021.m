%% Create fixed and moving arrays
rng(12345,'twister')
F=randn(50,1);
M=randn(50,1);
W=rand(50,1);
N=5;
eps=0.01;

% Exponent for the weight coefficient
p = 2;

% Create a fixed mask
K=zeros(50,1);
K(1:22)=1; K(33:45)=1;

fprintf('F = '); fprintf('%d, %d, %d, %d, %d, \n', F); fprintf('\n');
fprintf('M = '); fprintf('%d, %d, %d, %d, %d, \n', M); fprintf('\n');
fprintf('K = '); fprintf('%d, %d, %d, %d, %d, \n', K); fprintf('\n');
fprintf('W = '); fprintf('%d, %d, %d, %d, %d, \n', W); fprintf('\n');


%% Compute the components of Weighted NCC

% Running sums
sum_W = conv(W, ones(N,1), 'same');
sum_WF = conv(W.*F, ones(N,1), 'same');
sum_WM = conv(W.*M, ones(N,1), 'same');
sum_WFM = conv(W.*F.*M, ones(N,1), 'same');
sum_WFF = conv(W.*F.*F, ones(N,1), 'same');
sum_WMM = conv(W.*M.*M, ones(N,1), 'same');

% Variances
var_F =  sum_W .* sum_WFF - sum_WF .* sum_WF + eps;
var_M =  sum_W .* sum_WMM - sum_WM .* sum_WM + eps;
cov_FM = sum_W .* sum_WFM - sum_WM .* sum_WF;

% Weight scaling
w_scale = (sum_W / N).^p;

% Coefficients
ncc_FM = K .* sign(cov_FM) .* cov_FM.^2 ./ (var_F .* var_M);
ncc_FM_scaled = w_scale .* ncc_FM;

%% Compute the derivative of the total weighted NCC with respect to M

% Define the Q quantities
Q_fm = abs(cov_FM) ./ (var_F .* var_M);
Q_f = ncc_FM ./ var_F;
Q_m = ncc_FM ./ var_M;

% Define the y quantities
y1 = sum_W .* Q_fm;
y2 = sum_W .* Q_f;
y3 = sum_W .* Q_m;
y4 = (sum_WM .* Q_m - sum_WF .* Q_fm);
y5 = (sum_WF .* Q_f - sum_WM .* Q_fm);
y6 = 2 * sum_WFM .* Q_fm - sum_WFF .* Q_f - sum_WMM .* Q_m;

% Convolve to get the z quantities
z1 = conv(w_scale .* K .* y1, ones(N,1), 'same');
z2 = conv(w_scale .* K .* y2, ones(N,1), 'same');
z3 = conv(w_scale .* K .* y3, ones(N,1), 'same');
z4 = conv(w_scale .* K .* y4, ones(N,1), 'same');
z5 = conv(w_scale .* K .* y5, ones(N,1), 'same');
z6 = conv(w_scale .* K .* y6 + p * ncc_FM_scaled ./ sum_W, ones(N,1), 'same');

% Define the derivatives wrt to M and W
DM = 2 * W .* (F .* z1 - M .* z3 + z4); 
DW = 2*(F.*z5 + M.*z4 + F.*M.*z1) - F.^2 .* z2 - M.^2 .* z3 + z6; 

%% Does this really yield derivatives?
eps2 = 0.0001;
DM_num = zeros(50,1);
DW_num = zeros(50,1);
for pt = 1:50
    M1 = M; M1(pt) = M(pt) - eps2;
    M2 = M; M2(pt) = M(pt) + eps2;
    ncc1 = sum(compute_wncc(W, F, M1, K, N, p, eps));
    ncc2 = sum(compute_wncc(W, F, M2, K, N, p, eps));
    DM_num(pt) = (ncc2-ncc1) / (2 * eps2);

    W1 = W; W1(pt) = W(pt) - eps2;
    W2 = W; W2(pt) = W(pt) + eps2;
    ncc1 = sum(compute_wncc(W1, F, M, K, N, p, eps));
    ncc2 = sum(compute_wncc(W2, F, M, K, N, p, eps));
    DW_num(pt) = (ncc2-ncc1) / (2 * eps2);
end

clf;
plot(DW, 'r');
hold on;
plot(DW_num, 'ro');

plot(DM, 'b');
plot(DM_num, 'bo');

fprintf('Max error DM: %f,  DW: %f\n', max(abs(DM - DM_num)), max(abs(DW - DW_num)));

%% Print expected gradients in Greedy
grad_M = conv(M, [1,-1], 'same');
grad_W = conv(W, [1,-1], 'same');
grad_MW = conv(W .* M, [1,-1], 'same');

% Calculate the weights of the gradients
D_Phi = grad_MW .* (DM ./ W) + grad_W .* (DW - M .* DM ./ W);

fprintf('Metric = '); fprintf('%d, %d, %d, %d, %d, \n', ncc_FM_scaled); fprintf('\n');
fprintf('D_phi = '); fprintf('%d, %d, %d, %d, %d, \n', D_Phi); fprintf('\n');


