function [K]=main(Or, MT, Dim, T0, N, D)

% preprocessing
[B_init, MO, K, ktov, M,enlarge_final,dict_final,MT,N,All_min_max,Orn_final,ktov_or,O_final]=generate_test_seq(Dim, T0, N, Or, MT);
% [B_init, MO, K, ktov, M,enlarge_final,dict_final,MT,N,All_min_max,Orn_final,ktov_or]=generate_test_seq_normal(Dim, T0, N, Or, MT);
% B_init            used for initialize observation probability matrix
% MO                multiple observation sequences
% MT                length for each observation sequence
% K                 discrete observable values into K intervals

% % initialize model parameters
[A, B, P, PI]=initialize(B_init, D, K, M);

% % implement HSMM
[A,B,PI,P,S_est0]=hsmm(A, B, D, K, M, MO, MT, N, P, PI);
% % [A, B, D, K, M, M0, N, P, PI]     trained model parameters

save('hsmm_parameter_normal.mat','A', 'B', 'P', 'PI', 'D', 'K', 'M','enlarge_final','dict_final','MO','MT','N','ktov','All_min_max','Orn_final','ktov_or','O_final');

% % load abnormal data
[MO_abnormal] = generate_abnormal_seq(Or_abnormal, MT_abnormal,T0_abnormal, N_abnormal,en,dict_final,K)

% % Calculate the maximum likelihood probability of the attack data
[Loglikelihood] = hsmm_likelihood(A, B, P, PI, D, K, M, MO_abnormal, MT_abnormal, N_abnormal)


% compare the obsv seqs with the estimated results 
cc=zeros(size(ktov));
cc(ktov>0)=1;
avgObsv = (B * ktov) ./ (B * cc);
for d=1:Dim
    Ts = 0;
    for on=1:N
        T=MT(on);           % the length of the n'th obs seq
        S=S_est0(1:T,on);	    % the n'th state sequence
        figure;
        hold on;
        plot([0;Or(Ts+1:Ts+T,d)]);
        plot([0;avgObsv(S,(d-1)*N+on)],'r');
        hold off;
        saveas(gcf,['./fig_mul_abnormal_47/','seq_',num2str(d),'_',num2str(on),'.jpg']);
        Ts = Ts + T;
    end
end