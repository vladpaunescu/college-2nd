function [patterns] = read_digits(m)
%read_digits Citeste primele m sabloane (m <=10)
    if m > 10
        m = 10
    end
    patterns = zeros(m,120);
    fid = fopen('digits','r');
    for i = 1:m
        l = 0;
        for j = 1:10
            for k = 1:12
                c = fread(fid,1,'uchar');
                l = l+1;
                if c == '_'
                    patterns(i,l) = -1;
                else
                    patterns(i,l) = 1;
                end
            end
            fread(fid,1,'uchar');
        end
        fgetl(fid);
    end
end

% Tudor Berariu
        
