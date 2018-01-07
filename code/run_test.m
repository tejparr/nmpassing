clear all; close all;

T = 50;
n_states = 10;
trans_p = 0.05;

[obs, states] = discrete_bernoulli_process(T, n_states, trans_p);

%belief propagation/ forward-backward algorithm
beliefs1 = bp_algo(obs, n_states, trans_p);
plot_beliefs(beliefs1, states, T);

%variational massage passing
beliefs2 = var_algo(obs, n_states, trans_p);
plot_beliefs(beliefs2, states, T);

%expectation propagation
%not implemented
%beliefs3 = ep_algo(obs, n_states);
%plot_beliefs(beliefs3, states);