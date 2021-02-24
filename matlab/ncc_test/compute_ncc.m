function ncc_FM = compute_ncc(F, M, N, eps)

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
%ncc_FM = cov_FM.^2 ./ (var_F .* var_M);
ncc_FM = sign(cov_FM) .* cov_FM .* cov_FM ./ (var_F .* var_M);

end

