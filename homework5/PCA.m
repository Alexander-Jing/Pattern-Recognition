load ORLData_25;
X = ORLData;
X = double(X);
[n,d] = size(X);

dim = 1;
labels = X(:, dim); %��ȡ������������ǩ
labels = floor(double(labels));

labels(1:n ,1) 
c = max(labels)
%X(:, dim) = []; % ��ȡ��������
clear ORLData;

% load vehicle;
% out = UCI_entropy_data.train_data;
% X = out'; 
% X = double(X);
% [n, d] = size(X); 
% dim = d;
% labels = X(:, dim); 
% labels = floor(double(labels)) % ��ȡ������������ǩ
% c = max(labels); % c = 4
% X(:, dim) = []; % ��ȡ��������
% clear UCI_entropy_data;
% clear out;