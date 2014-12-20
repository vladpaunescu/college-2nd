function [xs, ys, T] = generateDataset(n, dist, radius)

	xs = zeros(1, n);

	x1 = -radius / 2;
	x2 = radius / 2;


	angles = rand(1, n) * pi;

	angles(n/2+1 : n) = angles(n/2 + 1 : n) + pi;

	rs = (radius - dist) * rand(1, n) + dist;

	xs = rs .* cos(angles);
	xs(1 : n / 2) = xs(1 : n / 2) + x1;
	xs(n / 2 + 1 : n) = xs(n / 2 + 1 : n) + x2;


	ys = rs .* sin(angles);

	ys(1 : n / 2) = ys(1 : n /2) - 2 * dist/3;
    figure
	scatter(xs(1 : n/2), ys(1 : n / 2), 'b')
	hold on;
	scatter(xs(n/2 + 1 : n), ys(n / 2 + 1 : n), 'r')
    hold off;
	T = zeros(n, 2);
	T(n/2 + 1 : n, 2) = 1;
	T(1 : n / 2, 1) = 1;
end
