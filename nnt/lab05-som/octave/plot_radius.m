%% Tudor Berariu, 15 Aprilie 2013

radius_values = zeros(1,1000);
for i=1:1000
    radius_values(i) = radius(i, 1000, 20, 20);
end
plot(radius_values)
