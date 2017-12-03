%simple matlab animation of the shape 
% first load data.
% then run anim.

close all
figure
idxInM = 1;

for idx =1:50:length(t)
plot(x(idx,:), y(idx,:),'r-');
hold on

% plot the torsional spring locations.
plot(x(idx, 2:end-1), y(idx, 2:end-1), 'ro')

grid
hold off
%axis([min(min(x)) 1 -0.1 0.1])
%pbaspect([4 1 1 ])
axis([min(min(x)) max(max(x)) min(min(y)) max(max(y))])
M(idxInM) = getframe;
idxInM = idxInM + 1;
end

myVideo = VideoWriter(strcat(fname,'.avi'));
open(myVideo);
writeVideo(myVideo, M);
close(myVideo);