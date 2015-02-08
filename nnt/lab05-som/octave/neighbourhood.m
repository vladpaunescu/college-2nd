function [ mask ] = neighbourhood( x, y, radius, width, height )
    %NEIGHBOURHOOD Construiește o mască cu valori în [0,1] pentru vecinătate

    %% Tudor Berariu, 15 Aprilie 2013

    mask = zeros(height, width);
    %% Taskul 4: Calculul vecinătății
    %% Taskul 4: completați aici

    %% Taskul 4: ----------
    
    for i = 1:height
        for j = 1:width
            if abs(y - i) + abs(x - j) < radius
                mask(i,j) = 1;
        end
    end
end
