function print_digit(pattern)
%print_digit Afiseaza un pattern (120 valori -1,1)
    digit = reshape(pattern,12,10)';
    ascii_digit = char((digit - 1) / (-2) * '_' + (digit + 1) / 2 * 'x');
    disp(ascii_digit)
end

% Tudor Berariu