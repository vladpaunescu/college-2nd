function [ n_pattern ] = add_noise( pattern, noise )
%distort_pattern Adauga zgomot unui sablon
    noise_mask = (rand(size(pattern)) >= noise) * 2 - 1;
    n_pattern = noise_mask .* pattern;
end

% Tudor Berariu

