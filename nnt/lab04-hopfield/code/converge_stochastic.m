function [ result ] = converge_stochastic( weights, new_p )
%converge Incearca sa clasifice new_p
    result = new_p;
    [m, n] = size(new_p);
    x = new_p;
    iter = 0;
    for i = 1 : 100
        i = randi(n);
        x(i) = sign(weights(i,:) * x');
    end
    while 1
       prev_x  = x;
          i = randi(n);
          x(i) = sign(weights(i,:) * x');
       if prev_x == x
           disp(sprintf('Converged at %d\n', iter));
           break;
       end
       iter = iter+1;
    end 
    result = x;
    disp(sprintf('\nInput'));
    print_digit(new_p);
    disp(sprintf('\nPrediction'));

    print_digit(result);
end



% Tudor Berariu

