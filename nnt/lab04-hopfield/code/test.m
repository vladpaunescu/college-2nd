function  test()
%TEST Summary of this function goes here
%   Detailed explanation goes here
noise = [0, 0.1, 0.15, 0.2, 0.25, 0.3];
n = size(noise, 2);
acc = zeros(10, n);
for i = 1:10
    [weights, patterns ] = train(i)
    for j = 1:n
       acc(i, j) = compute_accuracy( weights, patterns, noise(j));
    end
end

disp(acc);

figure

title('Accuracy variation');
xlabel('learned patterns');
ylabel('accuracy');

hold on

for j = 1:n
    plot(1:10, acc(:,j));
end


end

