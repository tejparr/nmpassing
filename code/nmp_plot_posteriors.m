function nmp_plot_posteriors(HMM)
colormap gray
Nf  = numel(HMM.B);
VMP = HMM.VMP;
BP  = HMM.BP;
for i = 1:Nf
    subplot(3,Nf,i)
    imagesc(1-VMP.Qs{i})
    title(['Posterior belief (VMP) factor ' num2str(i)])
    subplot(3,Nf,Nf+i)
    imagesc(1-BP.Qs{i})
    title(['Posterior belief (BP) factor ' num2str(i)])
    subplot(3,Nf,2*Nf+i)
    plot(HMM.s{i},'.r','MarkerSize',20)
    axis ij
    title(['True state - factor ' num2str(i)])
end

