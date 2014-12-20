function [ms , C] = kMeans(X, K)
%KMEANS Summary of this function goes here
%   Detailed explanation goes here
    [N,D] = size(X);
    rinds = randperm(N);
    ms = X(rinds(1:K), :)
    C = zeros (N, 1);
    for i = 1:N
       C(i) = nearestC(X(i,:), ms, K);
    end
    changed = 1;
    iter = 1;
    while changed && iter < 100
       disp(sprintf('iter %d\n', iter));
       iter = iter + 1;
       ms = computeMs(C, X, K);
       oldC = C;
       for i = 1:N
            C(i) = nearestC(X(i,:), ms, K);
       end
       if oldC == C
           changed = 0;
       end
    end
    %ms
    %C
end

function C = nearestC(x, m, K)
    C = 1;
    %minDist = sqrt((x - m(1, :)) * (x - m(1, :))');
    [dist, midx] = min(pdist2(x, m));
    C = midx;
%     
%     for i = 2:K
%         dist = pdist2(x, m(i, :));
%         if dist < minDist
%            minDist = dist;
%            C = i;
%         end
%     end
end

function ms = computeMs(C, X, K)
    ms = zeros(K, size(X, 2));
    for i = 1:K
        count = sum(C == i);
        s = sum(X(C==i, :));
        ms(i, :)  = s / count;
    end
    ms;
end
