function t = perceptronRule()
% meas(150x4) speecies 150 x 1 cell 
load fisheriris;
set = unique(species);
rows = size(species, 1);
t = zeros(rows, 1);
x = meas;
% get number of features
[examples, featureCount] = size(x);
% initialize w
for i = 1:size(set, 1)
    t(find(ismember(species, set(i)))) = i;
end
% consider all possible scenarios one vs all
target = [];
for i = 1: size(set, 1)
    target = (t == i) - (t ~= i);
    [success, w] = runPerceptron(x, target);
  %  drawPlot(x, target, w);
    if success
        break;
    end 
end
disp(sprintf('Best match for weights %f', w));
drawPlot(x, target, w);

end

function [success, w] = runPerceptron(x, t)
   [examples, featureCount] = size(x);
   w = zeros(featureCount, 1);
   iterCount = 1;
   while mispredict(x, w, t) > 0 && iterCount < 1000
       disp(sprintf('Epoch %d', iterCount));
       iterCount = iterCount + 1;
       index = randi([1, examples], 1, 1);
       disp(index);
       y = sign(w' * x(index, :)');
       if t(index) ~= y
           % update the weights
           w = w + x(index, :)' * t(index);
       end
   end
   success = (mispredict(x, w, t) == 0);
end

function error = mispredict(x, w, t)
    error = 0;
    for i = 1:size(x,1)
        if sign(w' * x(i, :)') ~= t(i)
            error = error + 1;
        end
    end
    disp(sprintf('Error %d', error));
end

function drawPlot(x, target, w)
    figure;
    hold on;
    gscatter(x(:,1), x(:,2), target,'rgb','osd');
    line(x(:,1), (-(w(1) * x(:,1) + w(3) * x(:,3) ...
                 + w(4) * x(:,4))) / w(2));
             
    x_min = min(x(:, 1)) - 1;
    x_max = max(x(:, 1)) + 1;
    y_min = min(x(:, 2)) - 1;
    y_max = max(x(:, 2)) + 1;
    [xx, yy] = meshgrid(linspace(x_min, x_max, 1000), ...
        linspace(y_min, y_max, 1000));
    contourf(x(:,1), x(:,2), 
    hold off;
end