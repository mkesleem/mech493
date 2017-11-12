i_interest = 3/dt;
maxFx = max(Fx(i_interest:end));
minFx = min(Fx(i_interest:end));
F_scale = max(abs(maxFx),abs(minFx));

maxy = max(y(i_interest:end));
miny = min(y(i_interest:end));
y_scale = max(abs(maxy),abs(miny));

H = figure;
hold on
xlabel('t')
ylabel('Fx, y, ydot')
title_string = sprintf('Fx and y vs. Time [N = %d, dt = 10e-%d, Sp = %0.2f]', N, exponent, sp);
title(title_string)
plot(t,Fx/F_scale, 'r-')
plot(t,y(:,1)/y_scale', 'b-')
plot(t,ydot(:,1)/y_scale', 'k-')
axis([min(t) max(t) min(min(Fx/F_scale),min(y(:,1)/y_scale)), max(max(Fx/F_scale),max(y(:,1)/y_scale))])
plot([0,t(end)],[0,0],'g')
legend('Fx/Fmax', 'y/ymax','ydot/ymax','0','Location','EastOutside');
fname = sprintf('Fx_y_Sp%.2f_N%d_dt10e-%d.fig',sp,N,exponent);
savefig(H,fname);