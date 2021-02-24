%% Create fixed and moving arrays
rng(12345,'twister')
F=randn(50,1);
M=randn(50,1);
W=rand(50,1);
N=5;
eps=0.01;

fprintf('F = '); fprintf('%d, %d, %d, %d, %d, \n', F); fprintf('\n');
fprintf('M = '); fprintf('%d, %d, %d, %d, %d, \n', M); fprintf('\n');
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

% Coefficients
ncc_FM = cov_FM.^2 ./ (var_F .* var_M);

%% Compute the derivative of the total weighted NCC with respect to M

% Define the Q quantities
Q1 = cov_FM ./ (var_F .* var_M);
Q2 = ncc_FM ./ var_F;
Q3 = ncc_FM ./ var_M;

% Define the y quantities
y1 = sum_W .* Q1;
y2 = sum_W .* Q2;
y3 = sum_W .* Q3;
y4 = (sum_WM .* Q3 - sum_WF .* Q1);
y5 = (sum_WF .* Q2 - sum_WM .* Q1);
y6 = 2 * sum_WFM .* Q1 - sum_WFF .* Q2 - sum_WMM .* Q3;

% Convolve to get the z quantities
z1 = conv(y1, ones(N,1), 'same');
z2 = conv(y2, ones(N,1), 'same');
z3 = conv(y3, ones(N,1), 'same');
z4 = conv(y4, ones(N,1), 'same');
z5 = conv(y5, ones(N,1), 'same');
z6 = conv(y6, ones(N,1), 'same');

% Define the derivatives wrt to M and W
DM = 2 * W .* (F .* z1 - M .* z3 + z4); 
DW = 2*(F.*z5 + M.*z4 + F.*M.*z1) - F.^2 .* z2 - M.^2 .* z3 + z6; 

%% Does this really yield derivatives?
eps2 = 0.0001;
DM_num = zeros(50,1);
DW_num = zeros(50,1);
for p = 1:50
    M1 = M; M1(p) = M(p) - eps2;
    M2 = M; M2(p) = M(p) + eps2;
    ncc1 = sum(compute_wncc(W, F, M1, N, eps));
    ncc2 = sum(compute_wncc(W, F, M2, N, eps));
    DM_num(p) = (ncc2-ncc1) / (2 * eps2);

    W1 = W; W1(p) = W(p) - eps2;
    W2 = W; W2(p) = W(p) + eps2;
    ncc1 = sum(compute_wncc(W1, F, M, N, eps));
    ncc2 = sum(compute_wncc(W2, F, M, N, eps));
    DW_num(p) = (ncc2-ncc1) / (2 * eps2);
end

clf;
plot(DW, 'r');
hold on;
plot(DW_num, 'ro');

plot(DM, 'b');
plot(DM_num, 'bo');

fprintf('Max error DM: %f,  DW: %f\n', max(abs(DM - DM_num)), max(abs(DW - DW_num)));