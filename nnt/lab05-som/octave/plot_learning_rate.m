%% Tudor Berariu, 15 Aprilie 2013

lr_values = zeros(1,1000);
for i=1:1000
    lr_values(i) = learning_rate(i, 1000);
end
plot(lr_values)
