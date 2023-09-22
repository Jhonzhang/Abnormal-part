function [A,B,PI,P, S_est0] = hsmm_2c(A, B, D, K, M, MO, MT, N, P, PI)
    % Author: Shun-Zheng Yu
    % Available: http://sist.sysu.edu.cn/~syu/Publications/hsmm_m.txt
    %
    % HSMM solves three fundamental problems for Hidden Semi-Markov Model (or explicit duration hidden
    % Markov model) using a new Forward-Backward algorithm published in
    %    IEEE Signal Processing Letters, Vol. 10, no. 1, pp. 11-14, January 2003: 
    %   ��An Efficient Forward-Backward Algorithm for an Explicit Duration Hidden Markov Model,��  
    %    by  S-Z. Yu and H. Kobayashi            
    % 
    %  This program is free software; you can redistribute it and/or
    %  modify it under the terms of the GNU General Public License
    %  as published by the Free Software Foundation; either version
    %  2 of the License, or (at your option) any later version.
    %  http://www.gnu.org/licenses/gpl.txt
    %
    % Update on Nov. 2014
    %
    %++++++++ The hidden semi-Markov model +++++++++++
    % M               Total number of states
    % N               Number of observation sequences
    % MT              Lengths of the observation sequences: MT=(T_1,...,T_N)
    % MO              Set of the observation sequences: MO=[O^1,...,O^N], O^n is the n'th obs seq.
    % D               The maximum duration of states
    % K               The total number of observation values
    % PI(1:M,1:1)    the initial probability of states
    %
    % A(1:M,1:M)      The state Transition Probability Matrix
    % for r=1:M
    %     A(r,r)=0;
    % end
    %
    % P(1:M,1:D);      The probability of state duration
    %
    % B(1:M,1:K)       The observation probability distribution
    %
    
    for ir=1:500      % Number of iterations
        A_est=zeros(M,M);
        P_est=zeros(M,D);
        B_est=zeros(M,K);
        PI_est=zeros(M,1);
        % S_est0=zeros(size(MO));
        S_est0=cell(1,size(MO,2));
        lkh=0;
        for on=1:N	        % for each observation sequence
            T=MT(on);       % the length of the n'th obs seq
            % O=MO(:,on);	    % the n'th observation sequence
            O=MO{1,on};	    % the n'th observation sequence
            %
            %++++++++++++++++++     Forward     +++++++++++++++++
            %---------------    Initialization    ---------------
            ALPHA=zeros(M,D+1);         			% the forward variable
            ALPHA(:,1:D)=repmat((PI.*B(:,O(1))),1,D).*P;	% Equ.(3)
            c=ones(T,1);
            s_ALPHA = sum(sum(ALPHA));
            s_ALPHA(s_ALPHA ==0) = 1;
            c(1) = 1 / s_ALPHA ; % scaling factor to avoid possible underflows
            % c(isnan(c)) = 0;
            S_est=zeros(T,1);
            ALPHA=ALPHA.*c(1);
            ALPHAm1=zeros(M,T);
            ALPHAm1(:,1)=ALPHA(:,1);
            ALPHAm1Amn=zeros(M,T);
            %---------------    Induction    ---------------
            for t=2:T
                ALPHAm1Amn(:,t-1)=(ALPHA(:,1)'*A)';			
                EL=repmat((ALPHAm1Amn(:,t-1).*B(:,O(t))),1,D);
                EL=EL.*P;
                ALPHA(:,1:D)=ALPHA(:,2:D+1).*repmat(B(:,O(t)),1,D)+EL;	%Equ.(2)
                s_ALPHA = sum(ALPHA(:));
                s_ALPHA(s_ALPHA ==0) = 1;
                c(t) = 1 / s_ALPHA;
                % c(isnan(c)) = 0;

                ALPHA=ALPHA.*c(t);
                ALPHAm1(:,t)=ALPHA(:,1);
            end

            c(c <= 0) = 1 + exp(-16);
            lkh=lkh-sum(log(c));       % the log likelihood
            % fprintf("%d,c(t):%d\n",ir,c);
            %++++++++ Backward and Parameter Re-estimation ++++++++
            %---------------    Initialization    ---------------
            BETA=ones(M,D);         % the backward variable and Equ.(7)
            GAMMA0=sum(ALPHA,2);	%Equ.(13)
            GAMMAmn=zeros(M,1);
            GAMMAnm=zeros(M,1);
            B_est(:,O(T))=B_est(:,O(T))+GAMMA0;
            [X,S_est(T)]=max(GAMMA0);
            %---------------    Induction    ---------------
            for t=(T-1):-1:1
                bm=B(:,O(t+1)).*c(t+1);
                EB=sum((P.*BETA),2);
                EB=bm.*EB;
                %% for estimate of A
                ROU=(ALPHAm1(:,t)*EB').*A;
                A_est=A_est+ROU;						%Equ.(8) for ZETA_{t+1}(m,n)
                %% for estimate of P
                P_est=P_est+repmat((ALPHAm1Amn(:,t).*bm),1,D).*P.*BETA;	%Equ.(9) for ETA_{t+1}(m,d)
                %% for estimate of state at time t
                GAMMAmn=GAMMAmn+sum(ROU,2);
                GAMMAnm=GAMMAnm+sum(ROU,1)';
                GAMMA=GAMMA0+GAMMAmn-GAMMAnm;   %Equ.(12) for GAMMA_t(m)
                I=find(GAMMA<0);			    %Due to the calculation precision, 
                                                %GAMMAmn-GAMMAnm may introduce a 
                                                %very small negtive number. 
                GAMMA(I)=0;
                s_GAMMA = sum(GAMMA);
                s_GAMMA(s_GAMMA==0) =1;
                GAMMA=GAMMA./s_GAMMA;
                [X,S_est(t)]=max(GAMMA);
                %% for estimate of B
                B_est(:,O(t))=B_est(:,O(t))+GAMMA;
                %% for update of backward variable
                BETA(:,2:D)=repmat(bm,1,D-1).*BETA(:,1:D-1);    %Equ.(5)
                BETA(:,1)=A*EB;						            %Equ.(6)
            end
            bm=B(:,O(1)).*c(1);
            P_est=P_est+repmat((PI.*bm),1,D).*P.*BETA;			%Equ.(9) for t=1.
                                            %i.e.,ETA_1(m,d)=ALPHA_1(m,d).*BETA_1(m,d)
            s_GAMMA = sum(GAMMA);
            s_GAMMA(s_GAMMA==0) =1;
            PI_est=PI_est+GAMMA./s_GAMMA;
            % S_est0(1:T,on)=S_est(1:T);
            S_est0{1,on}=S_est(1:T);
    
        end							% End for multiple observation sequences
        A0=A;
        B0=B;
     
        P0=P;
        PI0=PI;
        lkh0=lkh;
        % [ir, lkh] % each iteration the likelihood
        s_B_est = B_est;
        s_B_est = sum(B_est,2);
        s_B_est(s_B_est == 0) = 1;
        B_est=B_est./repmat(s_B_est,1,K);
    
        PI=PI_est./sum(PI_est);
        A=A_est./repmat(sum(A_est,2),1,M);
        s_B_est = B_est;
        s_B_est = sum(B_est,2);
        s_B_est(s_B_est == 0) = 1;
        B=B_est./repmat(s_B_est,1,K);
        P=P_est./repmat(sum(P_est,2),1,D);
                        
        %++++++++ To check if the model is improved ++++++++
        
        if ir>1
            % [ir, lkh-lkh1,log(1.01)]          % show improvement of log-likelihood
            % [ir, lkh-lkh1] 
            if (lkh-lkh1)<log(1.01)
                break
            elseif isnan(lkh-lkh1)
                % exit
                % lkh = 0
                % continue
                break
            end
        end
        %++++++++++++++++++++++++++++++++++++++++++++++++++++
        lkh1=lkh;
    end								% End for iteration.
    
    % save the trained model parameters if required
    Loglikelihood=lkh0;
    A0(isnan(A0)) = 0;
    B0(isnan(B0)) = 0;
    P0(isnan(P0)) = 0;
    PI0(isnan(PI0)) = 0;
    A=A0;
    B=B0;
    % hasNaN = any(isnan(B0), 'all');
    % % 输出判定结果
    % if hasNaN
    %     disp('hsmm.m :B0 Yes');
    % % else
    % %     disp('B0 No');
    % end
    P=P0;
    PI=PI0;
    
    