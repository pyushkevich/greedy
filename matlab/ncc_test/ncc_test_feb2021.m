%% Create fixed and moving arrays
rng(12345,'twister')
F=rand(50,1);
M=rand(50,1);
N=5;
eps=0.01;

fprintf('F = '); fprintf('%d, %d, %d, %d, %d, \n', F); fprintf('\n');
fprintf('M = '); fprintf('%d, %d, %d, %d, %d, \n', M); fprintf('\n');

%% Compute the components of NCC

% Running sums
sum_F = conv(F, ones(N,1), 'same');
sum_M = conv(M, ones(N,1), 'same');
sum_FM = conv(F.*M, ones(N,1), 'same');
sum_FF = conv(F.*F, ones(N,1), 'same');
sum_MM = conv(M.*M, ones(N,1), 'same');

% Variances
var_F = N * sum_FF - sum_F .* sum_F + eps;
var_M = N * sum_MM - sum_M .* sum_M + eps;
cov_FM = N * sum_FM - sum_M .* sum_F;

% Coefficients
ncc_FM = sign(cov_FM) .* cov_FM .* cov_FM ./ (var_F .* var_M);

%% Compute the derivative of the total NCC with respect to M

y1 = N * abs(cov_FM) ./ (var_F .* var_M);
z1 = conv(y1, ones(N,1), 'same') .* F;

y2 = - N * ncc_FM ./ var_M;
z2 = conv(y2, ones(N,1), 'same') .* M;

y3 = - (abs(cov_FM) ./ (var_F .* var_M)) .* sum_F + (ncc_FM ./ var_M) .* sum_M;
z3 = conv(y3, ones(N,1), 'same');

Dm = 2 * (z1+z2+z3);


%% Does this really yield derivatives?
eps2 = 0.0001;
d_num = zeros(50,1);
for p = 1:50
    M1 = M; M1(p) = M(p) - eps2;
    M2 = M; M2(p) = M(p) + eps2;
    ncc1 = sum(compute_ncc(F, M1, N, eps));
    ncc2 = sum(compute_ncc(F, M2, N, eps));
    d_num(p) = (ncc2-ncc1) / (2 * eps2);
end

clf;
plot(Dm, 'r');
hold on;
plot(d_num, 'ro');

%% What do we expect to get in Greedy?
grad_M = conv(M, [1,-1], 'same');

fprintf('Metric = '); fprintf('%d, %d, %d, %d, %d, \n', ncc_FM); fprintf('\n');
fprintf('D_phi = '); fprintf('%d, %d, %d, %d, %d, \n', Dm .* grad_M); fprintf('\n');
