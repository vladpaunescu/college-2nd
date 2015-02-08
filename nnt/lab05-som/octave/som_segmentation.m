function som_segmentation( orig_file_name, n )
    %SOM_SEGMENTATION Segmentează o imaginie utilizand SOM

    %% Tudor Berariu, 15 Aprilie 2013

    orig_img = imread(orig_file_name);
    orig_pixels = cast(orig_img,'double');
    orig_pixels = orig_pixels/255.0;
    X = orig_pixels;
    

    %% După rezolvarea Taskurilor 2, 3 și 4
    %% în fișierele learning_rate.m, radius.m și neighbourhood.m,
    %% rezolvați aici Taskurile 5 și 6.


    %% Taskul 5: antrenarea retelei Kohonen
    %% Taskul 5: completați aici
    W = rand(n,n,3) / 2;
    %% Taskul 5: ----------
    N = n * n % neurons
    tMax = 1000
    nSample = tMax
    [height, width, colors] = size(orig_pixels)
    randIDX = randi(width * height, nSample, 1); 
    for t = 1: tMax
       i = floor(randIDX(t)/width)  + 1;
       j = mod(randIDX(t), width) + 1;
       i, j
       x = squeeze(X(i, j, :));
       W = weight_update(x, W, t, tMax, width, height, n); 
    end
    W
      
 

    %% Taskul 6: compunerea imaginii segmentate pe baza ponderilor W
    %% Taskul 6: porniți de la codul din negative.m
    %% Taskul 6: completați aici

    % seg_pixels = TODO
    % seg_img = cast(seg_pixels,'uint8');
    % seg_file_name = TODO
    % image(seg_img)
    % imwrite(seg_img, seg_file_name);

    %% Taskul 6: ----------
    
    W
segment_p = segment_pixels(orig_pixels, W);
imshow(segment_p);

end

function [wm, ii, jj] = argmax(x, W, n)
    ii = -1;
    jj = -1;
    wm = intmin;
    x = squeeze(x);
    for i = 1:n
        for j = 1 : n
            v = squeeze(W(i,j,:));
            p = x' * v;
            if p > wm
                wm = p;
                ii = i;
                jj = j;
            end
        end
    end
end

function W = weight_update(x, W, t, tMax, width, height, n)
      [wm, imax, jmax] = argmax(x, W, n);
      eta =  learning_rate(t, tMax);
      r = radius( t, tMax, n, n );
      mask = neighbourhood(jmax, imax, r, n, n);
      for i = 1 : n
          for j = 1: n
              wij = squeeze(W(i,j,:));
              x
              wij
            W(i,j, :) = wij + eta * mask(i,j) * (x - wij); 
          end
      end
end

function seg_pixels = segment_pixels(orig_pixels, W)

   seg_pixels = orig_pixels;
   [height, width, colors] = size(orig_pixels);
   n = size(W, 1);
   W(1,1,:);
   for i = 1:height
       for j = 1:width
          % disp(sprintf('%d %d', i, j));
           x = squeeze(orig_pixels(i,j,:));
           [wm, imax, jmax] = argmax(x, W, n);
           imax, jmax
           seg_pixels(i,j,:) = squeeze(W(imax, jmax, :));
       end
    end
end