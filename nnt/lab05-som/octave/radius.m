function [ r ] = radius( iter_no, iter_count, width, height )
    %RADIUS Întoarce valoarea razei (vecinătății) pentru iterația curentă

    %% Tudor Berariu, 15 Aprilie 2013

    %% Taskul 3: calculul razei în funcție de dimensiunile rețelei și
    %%           de iterația curentă
    %% Taskul 3: completați aici

    %% Taskul 3: ----------
    upper= max(width, height) / 2;
    offset = 0;
    r = upper - (iter_no / iter_count) * upper + offset ;
    

end
