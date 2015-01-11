function [ acc ] = compute_accuracy( weights, patterns, noise )
%compute_accuracy Calculeaza acuratetea unei retele Hopfield
    m = size(patterns,1);
    accs = zeros(1,30);
    for i = 1:30
        digit = patterns(ceil(rand()*m),:);
        digit_n = add_noise(digit, noise);
        result = converge(weights, digit_n);
        accs(i) = mean(mean(digit == result));
    end
    acc = mean(accs);
end

% Tudor Berariu
