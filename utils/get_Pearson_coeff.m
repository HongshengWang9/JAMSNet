function Pearson_coeff = get_Pearson_coeff(X, Y)

% 本函数实现了皮尔逊相关系数的计算操作  Pearson_correlation_coefficient
%
% 输入：
%   X：输入的数值序列 
%   Y：输入的数值序列
%
% 输出：
%   Pearson_coeff：两个输入数值序列X，Y的皮尔逊相关系数
%


if length(X) ~= length(Y)
    error('两个数值数列的维数不相等');
end
% %%%%%%两种计算公式等价%%%%
% 第一种计算公式
x_m = mean(X);
y_m = mean(Y);
fenzi = sum((X-x_m).*(Y-y_m));
fenmu = sqrt(sum((X-x_m).^2)) .* sqrt(sum((Y-y_m).^2));
% 第二种计算公式
% fenzi = sum(X .* Y) - (sum(X) * sum(Y)) / length(X);
% fenmu = sqrt((sum(X .^2) - sum(X)^2 / length(X)) * (sum(Y .^2) - sum(Y)^2 / length(X)));
Pearson_coeff = fenzi / fenmu;
end

