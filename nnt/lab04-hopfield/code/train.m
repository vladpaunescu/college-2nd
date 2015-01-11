function [weights, patterns ] = train(m)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here
    [patterns] = read_digits(m);
    [m, n] = size(patterns);
    for i = 1:m
       print_digit(patterns(i, :));
    end
    [ weights ] = compute_weights(patterns);

end

