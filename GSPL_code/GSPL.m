
function [opt_index,W,obj] = GSPL(X,Y,feanum,m,knn_num)
% Input
% X: multi-view data
% Y: ground truth
% feanum: number of selected features
% m: reduced dimension
% knn_num: number of nearest neighbors

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix with k nonzero rows
% obj: objective function value

NITER = 20;
numview = length(X); 
N = size(X{1},1);
classnum = length(unique(Y));

%% ===================== Initialization =====================

% construct U
U = cell(numview,1);
for v = 1:numview
    S = constructW_PKN(X{v}',knn_num);
    Lv = Lss(S);
    eigvec = eig1(full(Lv),classnum+1,0);
    U{v} = eigvec(:,2:classnum+1);
    U{v} = U{v}./repmat(sqrt(sum(U{v}.^2,2)),1,classnum); 
end  

% initialize p
p0 = orth(rand(numview,1));

% initialize q
q0 = orth(rand(numview,1));

% initialize W
I = eye(N);
e = ones(N,1);
H = I-(e*e')/N;
HUv = cell(numview,1); 
sum_quut = zeros(N,N);
for v = 1:numview
    HUv{v} = H*U{v};
    sum_quut = sum_quut + q0(v)*HUv{v}*HUv{v}';
end
tem = cell(numview,1);
dv = zeros(numview,1);
conX = [];
for v = 1:numview
    tem{v} = p0(v)*X{v};
    conX = [conX,tem{v}];
    dv(v) = size(X{v},2);
end
dim = size(conX,2);
temp0 = conX'*sum_quut*conX;
lambda = 0.1; 
temp = temp0 + lambda*eye(dim);  % to guarantee that temp is positive semi-definite
W0 = orth(rand(dim,m));
[opt_index,W] = L20sparse(temp,feanum,W0);
W0 = W;

%% =====================  updating =====================

for iter = 1:NITER   
    
% calculate obj
obj(iter) = trace(W'*temp0*W);
if iter>2 && abs(obj(iter-1)-obj(iter))<10^-4
   break;
end

% update p
Wv = cell(numview,1);
Wv{1} = W(1:dv(1),:);
for j = 2:numview
    n = 0;
    for i = 1:(j-1)
        n = n + dv(i);
    end
    Wv{j} = W(n+1:n+dv(j),:);
end
A = cell(numview,1);
B = cell(numview,1);
for v = 1:numview
    A{v} = X{v}*Wv{v}; 
    B{v} = sum_quut*A{v};
end

AA = [];
BB = [];
for v = 1:numview
    av = [];
    bv = [];
    for k = 1:m
        av = [av; A{v}(:,k)];
        bv = [bv; B{v}(:,k)]; 
    end   
    AA = [AA, av];
    BB = [BB, bv];
end
ATB = AA'*BB;
p = eig1(ATB,1);

% update q
sum_phxw = zeros(N,m);
for v = 1:numview
    sum_phxw = sum_phxw + p(v)*A{v};
end
tra = zeros(numview,1);
sum_tra = 0;
for v = 1:numview
    tra(v) = trace(sum_phxw*sum_phxw'*HUv{v}*HUv{v}');
    sum_tra = sum_tra + tra(v)^2;
end
q = zeros(numview,1);
for v = 1:numview
    q(v) = tra(v)*((sum_tra)^(-0.5));
end

% update W
sum_quut = zeros(N,N);
for v = 1:numview
    sum_quut = sum_quut + q(v)*HUv{v}*HUv{v}';
end
tem = cell(numview,1);
conX = [];
for v = 1:numview
    tem{v} = p(v)*X{v};
    conX = [conX,tem{v}];
end
temp0 = conX'*sum_quut*conX;
temp = temp0 + lambda*eye(dim);
[opt_index,W] = L20sparse(temp,feanum,W0);
W0 = W;

end

plot(obj);

end


















