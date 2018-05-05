function nmp_plot_dynamics(HMM)
% M = [];

Xv = HMM.VMP.Xq;
Xb = HMM.BP.Xq;

V = [];
B = [];
for f = 1:numel(Xv)
Vf{f} = [];
Bf{f} = [];
        for j = 1:size(Xv{f},3)
            for k = 1:size(Xv{f},4)
                v = Xv{f}(:,:,j,k);
                Vf{f}(end+1,:) = v(:);
                b = Xb{f}(:,:,j,k);
                Bf{f}(end+1,:) = b(:);
                clear v b
            end
        end
V(:,end+1:end+size(Vf{f},2)) = Vf{f};
B(:,end+1:end+size(Bf{f},2)) = Bf{f};
end
PV = pca(V);
PV1 = V*PV(:,1);
PV2 = V*PV(:,2);
PV3 = V*PV(:,3);

PB = pca(B);
PB1 = V*PB(:,1);
PB2 = V*PB(:,2);
PB3 = V*PB(:,3);

for i = 1:length(V)
    subplot(2,2,1)
    plot(1:i,V(1:i,:))
    title('Beliefs (VMP)')
    axis([0 length(V) 0 1])
    
    subplot(2,2,2)
    plot(1:i,B(1:i,:))
    title('Beliefs (BP)')
    axis([0 length(B) 0 1])
    
    subplot(2,4,5)
    plot(PV1(1:i),PV2(1:i))
    xlabel('PC 1')
    ylabel('PC 2')
    axis([min(PV1) max(PV1) min(PV2) max(PV2)]);
    axis square
    
    subplot(2,4,7)
    plot(PB1(1:i),PB2(1:i))
    xlabel('PC 1')
    ylabel('PC 2')
    axis([min(PB1) max(PB1) min(PB2) max(PB2)]);
    axis square
    
    subplot(2,4,6)
    plot(PV2(1:i),PV3(1:i))
    xlabel('PC 2')
    ylabel('PC 3')
    axis([min(PV2) max(PV2) min(PV3) max(PV3)]);
    axis square
    
    subplot(2,4,8)
    plot(PB2(1:i),PB3(1:i))
    xlabel('PC 2')
    ylabel('PC 3')
    axis([min(PB2) max(PB2) min(PB3) max(PB3)]);
    axis square
    drawnow
    
%     if numel(M)
%         M(end + 1) = getframe(gcf);
%     else
%         M = getframe(gcf);
%     end
%     im = frame2im(M(end));
%     [A,map] = rgb2ind(im,256);
%     if i==1
%         imwrite(A,map,'C:\Users\Thomas\Dropbox\Code\Neuronal message passing\Dynamics.gif','gif','LoopCount',Inf,'DelayTime',0.1);
%     else
%         imwrite(A,map,'C:\Users\Thomas\Dropbox\Code\Neuronal message passing\Dynamics.gif','gif','WriteMode','append','DelayTime',0.1);
%     end
end
