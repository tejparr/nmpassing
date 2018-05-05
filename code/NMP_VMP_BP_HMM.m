function HMM = NMP_VMP_BP_HMM
% This demo defines a hidden Markov model, generates data from it, then
% inverts this model using variational message passing and belief
% propagation. For biological plausibility, gradient schemes are used.
% HMM specification:
% A{g}(g,s{1},s{2},s{3},s{4},s{5}) - likelihood for outcome modality g
% B{f}(s{f},s{f})                  - transitions for state factor f
% D{f}(s{f})                       - prior for intial state (factor f)
% T                                - time steps

rng default

% Define generative model
%==========================================================================
% Likelihood matrix
%--------------------------------------------------------------------------
d = 3; % dimensionality of hidden states
for i=1:d
    A{1}(:,:,i) = .5*eye(d)+.5*ones(d)/d;
end
% A{2}(:,:,1) = [1 0 0;
%                0 1 1];
% A{2}(:,:,2) = [1 0 0;
%                0 1 1];
% A{2}(:,:,3) = [1 0 0;
%                0 1 1];
           
% Transition probabilities
%--------------------------------------------------------------------------
if d == 3
    B{1} = [0.3 0.3  0.5;
            0   0.3  0.5;
            0.7 0.4  0 ];

    B{2} = [0.5 0.3  1;
            0   0.7  0;
            0.5 0    0];
else
    B{1} = 0.7*eye(d) + 0.3*(ones(d) -eye(d))/(d-1);
    B{2} = 0.9*eye(d) + 0.1*(ones(d) -eye(d))/(d-1);
end


% Prior probabilities
%--------------------------------------------------------------------------
if d == 3
    D{1} = [1 0 0]';
    D{2} = [0 0 1]';
else
    D{1} = zeros(d,1)/d;
    D{1}(1) = 1;
    D{2} = zeros(d,1);
    D{2}(end) = 1;
end

% Set up hmm structure
%--------------------------------------------------------------------------
%make true likelihood different from agents likelihood
diffA = 0;
diffB = 0;
diffD = 0;
if diffA
    for i=1:d
        TA{1}(:,:,i) = 0.9*eye(d)+0.1*ones(d)/d;
    end
else
    TA = A;
end

if diffB
   TB{1} = eye(d);
   TB{2} = B{2};
else
    TB = B;
end

if diffD
    TD = D;
    TD{1}(1) = 0;
    TD{1}(3) = 1;
else
    TD = D;
end

hmm.A = TA; % Likelihood
hmm.B = TB; % Transitions
hmm.D = TD; % Priors
hmm.T = 15; % Time

% Invert
%--------------------------------------------------------------------------
HMM = NMP_HMM_GP(hmm); % Generate data using HMM generative process
HMM.B = B;
HMM.A = A;
HMM.D = D;
VMP = NMP_VMP_HMM(HMM);% Invert using variational message passing
BP  = NMP_BP_HMM(HMM); % Invert using belief propagation

HMM.VMP = VMP;
HMM.BP  = BP;

% Figures
%--------------------------------------------------------------------------
% figure('Name','Posterior beliefs','Color','w')
% nmp_plot_posteriors(HMM)

figure('Name','Belief updating','Color','w','Position',[400 50 600 590])
nmp_plot_updates(HMM)

% figure('Name','Belief dynamics','Color','w','Position',[60 50 1200 590])
% nmp_plot_dynamics(HMM)

figure('Name','Free energy','Color','w','Position',[400 50 600 590])
plot(HMM.VMP.F),hold on
plot(HMM.BP.F)
legend('VMP Marginals','BP Marginals')
title('Variational free energy')
end

function HMM = NMP_HMM_GP(hmm)
% This function takes a hidden Markov model and uses it to generate data
% The HMM should have the following fields:
% A - likelihood matrix
% B - transition matrix
% D - prior over initial states
% T - length of data vector

A = hmm.A;
B = hmm.B;
D = hmm.D;
T = hmm.T;
for f = 1:numel(D)
    s{f}(1) = find(cumsum(D{f})>=rand, 1);
    for t = 2:T
        s{f}(t) = find(cumsum(B{f}(:,s{f}(t-1)))>=rand,1);
    end
end

for f = numel(D)+1:5
    s{f}(1:T) = 1;
end

for t = 1:T
    for g = 1:numel(A)
        o{g}(t) = find(cumsum(A{g}(:,s{1}(t),s{2}(t),s{3}(t),s{4}(t),s{5}(t)))>=rand,1);
    end
end

HMM = hmm;
HMM.s = s;
HMM.o = o;
end

function VMP = NMP_VMP_HMM(hmm)
% This function takes an HMM, and uses variational message passing to
% compute approximate posterior beliefs about the states, given
% sequentially presented outcomes. The HMM should have the following
% fields:
% A - likelihood matrix
% B - transition matrix
% D - prior over initial states
% o - data (optional)
% T - length of data vector

A = hmm.A;
B = hmm.B;
D = hmm.D;
o = hmm.o;
T = hmm.T;

% Initialisation
%--------------------------------------------------------------------------
for f = 1:numel(D)
    Ns(f) = length(D{f});
    Qs{f} = ones(Ns(f),T)/Ns(f);
end

tau = 4;
Ni  = 16;
% Message passing
%--------------------------------------------------------------------------

for t = 1:T
    for i = 1:Ni
        for f = 1:numel(D)
            lnAo = zeros(size(Qs{f}));
            for tt = 1:T
                v = nmp_ln(Qs{f}(:,tt));
                if tt<t+1
                    for g = 1:numel(A)
                        lnA = permute(nmp_ln(A{g}(o{g}(tt),:,:,:,:,:)),[2 3 4 5 6 1]);
                        for fj = 1:numel(D)
                            if fj == f
                            else
                                lnAs = nmp_dot(lnA,Qs{fj}(:,tt),fj);
                                clear lnA
                                lnA = lnAs; clear lnAs
                            end
                        end
                        lnAo(:,tt) = lnAo(:,tt) + squeeze(lnA);
                    end
                end
                if tt == 1
                    lnD = nmp_ln(D{f});
                    lnBs = nmp_ln(B{f})'*Qs{f}(:,tt+1);
                elseif tt == T
                    lnBs = zeros(size(D{f}));
                    lnD  = nmp_ln(B{f})*Qs{f}(:,tt-1);
                else
                    lnD  = nmp_ln(B{f})*Qs{f}(:,tt-1);
                    lnBs = nmp_ln(B{f})'*Qs{f}(:,tt+1);
                end
                v = v + (lnD + lnBs + lnAo(:,tt) - v)/tau;
                Ft(tt,i,t,f) = Qs{f}(:,tt)'*(lnD + lnAo(:,tt) - log(Qs{f}(:,tt)));
                Qs{f}(:,tt) = exp(v)/sum(exp(v));
                Xq{f}(:,tt,t,i) = Qs{f}(:,tt);
                clear v
            end
        end
    end
end
F = sum(Ft,4);
F = squeeze(sum(F,1));

VMP    = hmm;
VMP.Qs = Qs; % Posteriors at end
VMP.Xq = Xq; % Posteriors throughout
VMP.F  = -F(:);% Free energy
end

function BP = NMP_BP_HMM(hmm)
% This function takes an HMM, and uses belief propagation to compute
% marginal beliefs about states, given serially presented outcomes. Unlike
% classical approaches to BP (e.g. Baum-Welch), we use a gradient ascent
% scheme that relies upon messages derived from posterior marginals
% The HMM should have the following fields:
% A - likelihood matrix
% B - transition matrix
% D - prior over initial states
% o - data (optional)
% T - length of data vector

A = hmm.A;
B = hmm.B;
D = hmm.D;
o = hmm.o;
T = hmm.T;

% Initialisation
%--------------------------------------------------------------------------
for f = 1:numel(D)
    Ns(f) = length(D{f});
    Qs{f} = ones(Ns(f),T)/Ns(f);
    Mf{f} = ones(Ns(f),T)/Ns(f); Mf{f}(:,1) = D{f};
    Mb{f} = ones(Ns(f),T)/Ns(f);
end

tau = 4;
Ni  = 16;
% Message passing
%--------------------------------------------------------------------------
for t = 1:T
    for i = 1:Ni
        for f = 1:numel(D)
            lnAo = zeros(size(Qs{f}));
            for tt = 1:T
                v = nmp_ln(Qs{f}(:,tt));
                if tt<t+1
                    for g = 1:numel(A)
                        Ao = permute(A{g}(o{g}(tt),:,:,:,:,:),[2 3 4 5 6 1]);
                        for fj = 1:numel(D)
                            if fj == f
                            else
                                As = nmp_dot(Ao,Qs{fj}(:,tt),fj);
                                clear Ao
                                Ao = As; clear As
                            end
                        end
                        lnAo(:,tt) = squeeze(nmp_ln(Ao)) + lnAo(:,tt);
                    end
                end
                
                % Update messages
                for ttt = 1:T
                    if ttt<t+1 && ttt<T
                        if ttt>1
                            vv = exp(nmp_ln(Qs{f}(:,ttt))-nmp_ln(Mb{f}(:,ttt))-lnAo(:,ttt));
                            Mf{f}(:,ttt) = vv/sum(vv);
                        end
                        vv = exp(nmp_ln(Qs{f}(:,ttt))-nmp_ln(Mf{f}(:,ttt))-lnAo(:,ttt));
                        Mb{f}(:,ttt) = vv/sum(vv);
                    end
                end
                % Update marginals
                if tt == 1
                    lnD = nmp_ln(D{f});
                else
                lnD  = nmp_ln(B{f}*Mf{f}(:,tt-1));
                end
                if tt<T
                lnBs = nmp_ln(B{f}'*Mb{f}(:,tt+1));
                else
                    lnBs = ones(size(Qs(:,1)));
                end
                v = v + (lnD + lnBs + lnAo(:,tt) - v)/tau;
                Ft(tt,i,t,f) = Qs{f}(:,tt)'*(lnD + lnAo(:,tt) - log(Qs{f}(:,tt)));
                Qs{f}(:,tt) = exp(v)/sum(exp(v));
                Xq{f}(:,tt,t,i) = Qs{f}(:,tt);
            end
        end
    end
end

F = sum(Ft,4);
F = squeeze(sum(F,1));

BP    = hmm;
BP.Qs = Qs;
BP.M.f = Mf;
BP.M.b = Mb;
BP.Xq  = Xq;
BP.F   = -F(:);
end

function y = nmp_ln(x)
% For numerical reasons
y = log(x+exp(-16));
end

function B = nmp_dot(A,s,f)
% multidimensional dot product along dimension f
d = zeros(1,5);
d(f) = 1;
for i = 2:5
    d(find(d==0,1))=i;
end
x = permute(s,d) + zeros(size(A));
B = sum(A.*x,f);
k = zeros(1,5);
k(f) = 5;
for i = 1:4
    k(find(k==0,1))=i;
end
B = permute(B,k);
end

