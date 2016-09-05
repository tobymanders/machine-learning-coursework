function D=l2distance(X,Z)
% function D=l2distance(X,Z)
%	
% Computes the Euclidean distance matrix. 
% Syntax:
% D=l2distance(X,Z)
% Input:
% X: dxn data matrix with n vectors (columns) of dimensionality d
% Z: dxm data matrix with m vectors (columns) of dimensionality d
%
% Output:
% Matrix D of size nxm 
% D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
%
% call with only one input:
% l2distance(X)=l2distance(X,X)
%

if (nargin==1) % case when there is only one input (X)
	[d,n] = size(X);
    m = 1;
    Z = zeros(d,1);
    
    S = repmat(diag(X'*X),1,m);
    R = repmat(diag(Z'*Z)',n,1);
    G = X'*Z;
    
    D = sqrt(S - 2*G + R);
else  % case when there are two inputs (X,Z)
	
    [~,n] = size(X);
    [~,m] = size(Z);
    
    S = repmat(diag(X'*X),1,m);
    R = repmat(diag(Z'*Z)',n,1);
    G = X'*Z;
    
    D = sqrt(S - 2*G + R);

%% One Line Version
%     D = sqrt(repmat(diag(X'*X),1,m) - 2*(X'*Z) + repmat(diag(Z'*Z)',n,1));

end;
