close all
%simple matlab animation of the shape 
% first load data.
% then run anim.


%set(h, 'Position', [100 100 400 400])
figure

for idx =1:1000:length(t)
plot(x(idx,:), y(idx,:),'r-');
hold on

% plot the torsional spring locations.
plot(x(idx, 2:end-1), y(idx, 2:end-1), 'ro')

grid
hold off
%axis([min(min(x)) 1 -0.1 0.1])
%pbaspect([4 1 1 ])
axis([min(min(x)) max(max(x)) min(min(y)) max(max(y))])
pause(dt)
end