function Pearson_coeff = get_Pearson_coeff(X, Y)

% ������ʵ����Ƥ��ѷ���ϵ���ļ������  Pearson_correlation_coefficient
%
% ���룺
%   X���������ֵ���� 
%   Y���������ֵ����
%
% �����
%   Pearson_coeff������������ֵ����X��Y��Ƥ��ѷ���ϵ��
%


if length(X) ~= length(Y)
    error('������ֵ���е�ά�������');
end
% %%%%%%���ּ��㹫ʽ�ȼ�%%%%
% ��һ�ּ��㹫ʽ
x_m = mean(X);
y_m = mean(Y);
fenzi = sum((X-x_m).*(Y-y_m));
fenmu = sqrt(sum((X-x_m).^2)) .* sqrt(sum((Y-y_m).^2));
% �ڶ��ּ��㹫ʽ
% fenzi = sum(X .* Y) - (sum(X) * sum(Y)) / length(X);
% fenmu = sqrt((sum(X .^2) - sum(X)^2 / length(X)) * (sum(Y .^2) - sum(Y)^2 / length(X)));
Pearson_coeff = fenzi / fenmu;
end

