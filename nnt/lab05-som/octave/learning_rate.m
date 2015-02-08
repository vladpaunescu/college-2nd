function [ lr ] = learning_rate( iter_no, iter_count )
    %LEARNING_RATE Întoarce valoarea coeficientului de învățare

    %% Tudor Berariu, 15 Aprilie 2013

    %% Taskul 2: calculul coeficientului de învățare
    %% Taskul 2: completați aici

    %% Taskul 2: ----------
    upper = 0.65;
    offset = 0.1;
    lr = upper - (iter_no / iter_count) * upper + offset ;
end
