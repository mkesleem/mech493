[m,n] = size(x);
distance_matrix = zeros(m,n-1);

for j = 1:m
    for i = 1:n-1
        point_vector = [x(j,i),y(j,i);x(j,i+1),y(j,i+1)];
        distance_matrix(j,i) = pdist(point_vector,'euclidean');
    end
end

fprintf('\nMax distance is %.2f \n', max(max(distance_matrix)));
fprintf('Min distance is %.2f \n', min(min(distance_matrix)));
fprintf('Expected distance is %.2f \n', 1/N);