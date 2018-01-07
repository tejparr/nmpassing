function plot_beliefs(beliefs, states, T)
    n = size(beliefs,1);
    T_total = size(beliefs,2);
    time = 1:T_total;
    CLIM = [0 1];
    p = linspace(1/(2*n), 1-1/(2*n), n)';
    
    figure();
    imagesc(time, p, beliefs, CLIM);
    colorbar()
    ylim([0,1])
    hold on
    plot(1:T, p(states)', 'wo');
    plot([T,T], [0,1], 'r');
    hold off;
end