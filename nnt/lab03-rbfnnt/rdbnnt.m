function [ preds, acc ] = rdbnnt(N, dist, radius, K, sigma)
close all;
disp(sprintf('Computing clusters'));
[ms, C, X, T] = twoMoons(N, dist, radius, K)
disp('Centers for radial basiss fucntions are')
disp(ms);
fis = zeros(N, K);
for  i = 1 : K
   fis(:, i) = gauss(X, ms(i,:), sigma);
end
disp(fis);

%surf(X(:,1),X(:, 2), fis(:, 1));


T;
W = pinv(fis) * T;

disp('Predicting')
[xs, ys, T2] = generateDataset(N, dist, radius);

ySize = size(T2, 2);
P = [xs', ys'];
count = 0;
preds = zeros(N, 1);
for i = 1:N
    p = predict(P(i,:), ms, W, K, sigma, ySize);
    pred = (p == max (p));
    %T2(i,:);
    if pred(1) == 1
        preds(i) = 1;
    end
    
    if pred == T2(i,:)
        count = count + 1;
    end
end
disp(sprintf('Prediction count %f\n',count / N));

acc = count/N;

figure;

c1 = X(preds == 0, :);
c2 = X(preds == 1, :);
scatter(c1(:,1), c1(:, 2), 'r');
hold on;
scatter(c2(:,1), c2(:, 2), 'b');
hold off;

end

function y = predict(p, ms, W, K, sigma, ySize)
    fis = gauss2(p, ms, sigma);
    %size(fis);
    %size (W);
    y = fis * W;
end
        

function fi = gauss(X, m, sigma)
    dists = pdist2(X, m).^ 2;
    div =  2 * (sigma^2);
    fi = exp (-dists / div);
end

function fis = gauss2(x, ms, sigma)
    dists = pdist2(x, ms).^ 2;
    div =  2 * (sigma^2);
    fis = exp (-dists / div);
end