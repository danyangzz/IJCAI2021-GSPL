function [opt_index,W] = L20sparse(A,k,W0)
% Input
% A: positive semi-definite matrix
% k: number of selected features
% W0: initialization of W

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix with k nonzero rows

NITER_W = 20;
[dim,m] = size(W0);

%% ==================== Initialization =================

W = W0;

%% =====================  updating =====================

for iter = 1:NITER_W
   P = A*W*pinv(W'*A*W)*W'*A;
   [~, ind] = sort(diag(P), 'descend');
   opt_index = sort(ind(1:k));
   Aopt = A(opt_index, opt_index);
   [V, ~] = eig1(Aopt, m);
   W = zeros(dim,m);
   W(opt_index, :) = V; 
end

end

