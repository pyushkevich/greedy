function ncc_FM = compute_wncc(W, F, M, N, eps)

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

end

