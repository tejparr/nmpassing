function nmp_plot_updates(HMM)
% M = [];

Xv = HMM.VMP.Xq;
Xb = HMM.BP.Xq;

Nf = numel(Xv);

for t = 1:HMM.T
    subplot(3,1,3)
    for g = 1:numel(HMM.o)
        plot(HMM.o{g}(1:t)+g/numel(HMM.o),'.','MarkerSize', 30), hold on
        axis([0 HMM.T+1 0 max(HMM.o{1})+1])
        axis ij
        ylabel('Sensory data')
    end
    hold off
    for i = 1:size(Xv{1},4)
        for f = 1:Nf
            subplot(3,Nf,f)
            imagesc(1-Xv{f}(:,:,t,i)), axis off
            title(['Posterior belief (VMP) factor ' num2str(f)])
            subplot(3,Nf,Nf+f)
            imagesc(1-Xb{f}(:,:,t,i)), axis off
            title(['Posterior belief (BP) factor ' num2str(f)])
        end
        colormap gray
        
%         if numel(M)
%             M(end + 1) = getframe(gcf);
%         else
%             M = getframe(gcf);
%         end
%         im = frame2im(M(end));
%         [A,map] = rgb2ind(im,256);
%         if t==1 && i == 1
%             imwrite(A,map,'C:\Users\Thomas\Dropbox\Code\Neuronal message passing\NMP.gif','gif','LoopCount',Inf,'DelayTime',0.1);
%         else
%             imwrite(A,map,'C:\Users\Thomas\Dropbox\Code\Neuronal message passing\NMP.gif','gif','WriteMode','append','DelayTime',0.1);
%         end
        pause(0.01)
    end
end