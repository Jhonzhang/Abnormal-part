function Loglikelihood = hsmm_likelihood_2f(A, B, P, PI, D, K, M, MO, MT, N,T0,real_MT)
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
    % format long g;
    lkh = 0;
    % format short
    lkh_array = [];
    B(isnan(B)) = exp(-16);
    % probMatrix = B;
    % hasNaN = any(isnan(probMatrix), 'all');
    % % 输出判定结果
    % if hasNaN
    %     disp('likelihood isnan:Yes');
    % % else
    % %     disp('likelihood isnan:No');
    % end
    % real_MT
    % disp(size(B));
    
    for on = 1:N % for each observation sequence
        % T = MT_abnormal(on); % the length of the n'th obs seq
        T = real_MT(on);
        O = MO{1, on}; % the n'th observation sequence
        % O = MO(:, on); % the n'th observation sequence
        % disp(size(O));
        % fprintf("size O:%d %d\n",size(O));
        % fprintf("size B:%d %d\n",size(B));
        % fprintf("size PI:%d %d\n",size(PI));
        % O(1)
        %
        %++++++++++++++++++     Forward     +++++++++++++++++
        %---------------    Initialization    ---------------
        ALPHA = zeros(M, D + 1); % the forward variable
        ALPHA(:, 1:D) = repmat((PI .* B(:, O(1))), 1, D) .* P; % Equ.(3)
        ALPHA(isnan(ALPHA)) = 0;
        c = ones(T, 1);
        % if any(isnan(ALPHA),'all')
        %     disp('1yes NaN!');
        % % else
        % %     disp('1No NaN!');
        % end
        s_ALPHA = sum(sum(ALPHA));
        % s_ALPHA(s_ALPHA ==0) = 1;
        % c(1) = 1 / s_ALPHA ; % scaling factor to avoid possible underflows
        if s_ALPHA == 0
            c(1) = 1;
        else
            % s_ALPHA(s_ALPHA ==0) = 1;
            c(1) = 1 / s_ALPHA ; % scaling factor to avoid possible underflows
        end
        % lkh = -sum(log(c)); % the log likelihood
        % lkh
        % S_est = zeros(T, 1);
        ALPHA = ALPHA .* c(1);
       
        ALPHAm1 = zeros(M, T);
        ALPHAm1(:, 1) = ALPHA(:, 1);
        ALPHAm1Amn = zeros(M, T);
        
        %---------------    Induction    ---------------
        for t = 2:T
            ALPHAm1Amn(:, t - 1) = (ALPHA(:, 1)' * A)';
            % ALPHAm1Amn(isnan(ALPHAm1Amn)) = 0;
            % t
            % size(B)
            % size(O(t));
            % size(O)
            % O(t)
            % ALPHAm1Amn(:, t - 1);
            % B(:, O(t));
            EL = repmat((ALPHAm1Amn(:, t - 1) .* B(:, O(t))), 1, D);
            EL = EL .* P;
            EL(isnan(EL)) = 0;
            ALPHA(:, 1:D) = ALPHA(:, 2:D + 1) .* repmat(B(:, O(t)), 1, D) + EL; %Equ.(2)
            ALPHA(isnan(ALPHA)) = 0;
            % if any(isnan(EL),'all')
            %     disp('EL yes NaN!');
            % % else
            % %     disp('ALPHA No NaN!');
            % end
            s_ALPHA = sum(ALPHA(:));
            % s_ALPHA(s_ALPHA ==0) = 1;
            if s_ALPHA == 0
                c(t) = 1;
            else
                % s_ALPHA(s_ALPHA ==0) = 1;
                c(t) = 1 / s_ALPHA ; % scaling factor to avoid possible underflows
            end
            % if any(isnan(ALPHA),'all')
            %     disp('2yes NaN!');
            % else
            %     disp('2No NaN!');
            % end
            % lkh = -sum(log(c)); % the log likelihood
            % lkh
            ALPHA = ALPHA .* c(t);
            ALPHAm1(:, t) = ALPHA(:, 1);
        end
        % disp(c');
        % probMatrix = c;
        % hasNaN = any(isnan(probMatrix), 'all');
        % % 输出判定结果
        % if hasNaN
        %     disp('c(t) isnan:Yes');
        % % else
        % %     disp('c(t) isnan:No');
        % end
        % c(isnan(c)) = 0;
        % c(c <= 0) = 1 + exp(-16);
        c(c == 1) = 1 + exp(-16);
        lkh = -sum(log(c)); % the log likelihood
        % lkh
        % T
        % disp(vpa(lkh/T));
        % lkh = lkh/T;
        lkh = double(vpa(lkh)/T);
        % lkh
        % format short
        % format(lkh)
        % % class(lkh)
        % lkh
        % a_t = round(lkh,3);
        % a_t
        % lkh
        % lkh = roundn(lkh,-3); % the log likelihood 2
        % lkh_array
        % lkh = sum(log(c);
        % [on, lkh] % show improvement of log-likelihood
       
        % fprintf("Abnormal data: this is the %d 'th seq, the likelihood is: %.3f\n",on, lkh);
       
        lkh_array = [lkh_array,lkh];
        lkh_array = round(lkh_array,3);
        % class(lkh_array)
        % break;

    end % End for multiple observation sequences

    % %++++++++ To check if the model is improved ++++++++

    %     if ir>1
    %         [ir, lkh-lkh1]          % show improvement of log-likelihood
    %         if (lkh-lkh1)<log(1.01)
    %             break
    %         elseif isnan(lkh-lkh1)
    %             exit
    %         end
    %     end
    %     %++++++++++++++++++++++++++++++++++++++++++++++++++++
    %     lkh1=lkh;
    % end								% End for iteration.

    % save the trained model parameters if required
    Loglikelihood = lkh_array;
    % disp(Loglikelihood);

