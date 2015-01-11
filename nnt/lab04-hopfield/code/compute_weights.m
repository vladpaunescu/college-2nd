function [ weights ] = compute_weights( patterns )
%compute_weights Calculeaza ponderile retelei Hopfield pentru n sabloane de
%dimensiune m
    [m, n] = size(patterns);
    weights = zeros(n,n);
    size(patterns)
    s = patterns;
    I = eye(n);
    
    for i = 1:m
%         for j = 1 : n
%             for k = 1:n
%            %    weights(j, k) = weights(j, k) + s(i, j) .* s(i,k);
%             end
%         end
        weights = weights + s(i,:)' * s(i,:);
    end
    
    weights = weights - m * I;    
    
end

% Tudor Berariu

