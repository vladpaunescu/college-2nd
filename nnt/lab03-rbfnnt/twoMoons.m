function  [ms, C, X, T] = twoMoons(n, dist, radius, K)
    [xs, ys, T] = generateDataset(n, dist, radius);
    X = [xs', ys'];
 
    [ms , C] = kMeans(X, K);
    colors = 'ymcrgbwk';
    colCount = size(colors,2);
    figure
    for i = 1:K
       ps = X(C == i, :)
       color = colors(mod(i,colCount) + 1);
       scatter(ps(:, 1), ps(:, 2), color);
       hold on;
       scatter(ms(i,1), ms(i,2), color, 'x');
    end
    hold off;
end

