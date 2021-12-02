load ORLData_25;
X = ORLData;
X = double(X);
[n,d] = size(X);

dim = 1;
labels = X(:, dim); %获取各样本的类别标签
labels = floor(double(labels));

labels(1:n ,1) 
c = max(labels)
%X(:, dim) = []; % 获取样本数据
clear ORLData;

% load vehicle;
% out = UCI_entropy_data.train_data;
% X = out'; 
% X = double(X);
% [n, d] = size(X); 
% dim = d;
% labels = X(:, dim); 
% labels = floor(double(labels)) % 获取各样本的类别标签
% c = max(labels); % c = 4
% X(:, dim) = []; % 获取样本数据
% clear UCI_entropy_data;
% clear out;