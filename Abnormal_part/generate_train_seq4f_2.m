function result = generate_train_seq4f_2(Dim, T0, N, Or, MT, D, Or_abnormal, MT_abnormal, Dim_abnormal, T0_abnormal, N_abnormal, D_abnormal,Or_test, MT_test,N_test)
    % Dim, T0, N, Or, MT, D, Or_test, MT_test, Dim_test, T0_test, N_test, D_test
    % Dim,T0,N_train,Or_train, MT_train,D,Or_abnormal, MT_abnormal, Dim_abnormal, T0_abnormal, N_abnormal, D_abnormal,Or_test, MT_test,N_test
    start_time = datetime('now');
    disp(['start time:',char(start_time)]);
    num_vars = 3; % 参数个数
    lb = [5,5,2];
%    ub = [150,50,20];
    M_max = 150;
    en = 50;
    r = 20;
    fprintf("The test parameters: M:%d,en:%d,r:%d\n", M_max, en,r);
    ub = [M_max,en,r];
    cnt_ga = 0;
    % intCon = 1:3;
    % ga_options = optimoptions('ga', 'MaxGenerations', 100, 'PopulationSize', 100,'TolFun', 5e-2);
    % ga_options = optimoptions('ga', 'MaxGenerations', 50,'TolFun', 5e-2);
    % ga_options = optimoptions('ga', 'MaxGenerations', 100,'MaxStallGenerations', 20,'PopulationSize', 100,'FitnessLimit',5e-2);
    % ga_options = optimoptions('ga', 'MaxGenerations', 100,'PopulationSize', 100,'FitnessLimit',5e-2);
    % ga_options = optimoptions('ga','FitnessLimit',5e-2);
    % ga_options = optimoptions('ga','FitnessLimit',5e-2);
    ga_options = optimoptions('ga','MaxGenerations', 500,'FitnessLimit',5e-2);
    % ga_options = optimoptions('ga','FitnessLimit',1e-2);
    % fitness_func = @(params) cal_key_parameters(Dim, T0, N, Or, MT, D, Or_test, MT_test, Dim_test, T0_test, N_test, D_test, params);
    integerStep = 1;
    fitness_func = @(params)cal_key_parameters(round(params / integerStep) * integerStep,0);
    intcon= [1,2,3];
    [best_params, best_fitness] = ga(fitness_func, num_vars, [], [], [], [], lb, ub, [],intcon, ga_options);
    % disp(['M:',num2str(best_params(1))]);
    % disp(['en:',num2str(best_params(2))]);
    % disp(['r:',num2str(best_params(3))]);
    disp(['best_fitness:',num2str(best_fitness)]);

    variance_normal= cal_key_parameters(round(best_params),1); %
    disp([variance_normal,best_params]);
    % format short
    if variance_normal <= 5e-2
        disp(['Yes!best!p:', num2str(variance_normal)]);
    else
        disp('No same!,but is not OK!');
    end
    result = best_fitness;

    end_time = datetime('now');
    elapsed_time = end_time - start_time;
    disp(['end time:',char(end_time)]);
    disp(['Running time:',char(elapsed_time)]);

    function variance_normal = cal_key_parameters(params,vertifly_flag)

        M = params(1);
        enlarge = params(2);
        r = params(3);
        en = enlarge;
        % use_times_cnt = use_times_cnt + 1;
        % use_parametes{use_times_cnt} = [M, en, K];
        [dict_final, K, All_min_max, enlarge_final, S_min_max, Orn_final,Orn_back,O] = discrete(en, Or,r);
        cnt_ga = cnt_ga + 1;
        if mod(cnt_ga,50) == 0
            fprintf("Cnt:%d, Defined parameters: M:%d,en:%d,k:%d,r:%d,M_max:%d\n",cnt_ga, M, en, K,r,M_max);
        end
    
        if M < K
            [ktov, B_init, MO_dist, MO, ktov_or] = init_some_parameter(K, Dim, N, M, O, MT, Or,Orn_final);
            % initialize model parameters
            [A_normal, B_normal, P_normal, PI_normal] = initialize(B_init, D, K, M);
            % implement HSMM
            [A, B, PI, P, S_est0] = hsmm_2c(A_normal, B_normal, D, K, M, MO, MT, N, P_normal, PI_normal);
            % test normal
            [MO_real_normal, real_MT_normal, smoothed_B_normal, map_percent_noraml] = generate_test_seq4d(Or, MT, T0, N, enlarge_final, dict_final, K, ktov, M, All_min_max, B, S_min_max, r);

            Loglikelihood_normal = hsmm_likelihood_2f(A, smoothed_B_normal, P, PI, D, K, M, MO_real_normal, MT, N, T0, real_MT_normal);
            % test abnormal
            variance_normal = var(Loglikelihood_normal);
            if vertifly_flag
                cnt_ga = cnt_ga - 1;
                fprintf("Cnt:%d, Defined parameters: M:%d,en:%d,k:%d,r:%d,M_max:%d\n",cnt_ga, M, en, K,r,M_max);
                % Or_abnormal, MT_abnormal, Dim_abnormal, T0_abnormal, N_abnormal
                [MO_real_abnormal, real_MT_abnormal, smoothed_B_abnormal, map_percent_abnormal] = generate_test_seq4d(Or_abnormal, MT_abnormal, T0_abnormal, N_abnormal, enlarge_final, dict_final, K, ktov, M, All_min_max, smoothed_B_normal, S_min_max, r);

                Loglikelihood_abnormal = hsmm_likelihood_2f(A, smoothed_B_abnormal, P, PI, D_abnormal, K, M, MO_real_abnormal, MT_abnormal, N_abnormal, T0_abnormal, real_MT_abnormal);
                % format short
                mean_normal = mean(Loglikelihood_normal);
                std_normal = std(Loglikelihood_normal);
                Loglikelihood_normal_cal = round(abs(Loglikelihood_normal - mean_normal)/std_normal,3);
                Loglikelihood_abnormal_cal = round(abs(Loglikelihood_abnormal - mean_normal)/std_normal,3);
                variance_abnormal = var(Loglikelihood_abnormal);
                
                % [h, p, ksstat] = kstest2(Loglikelihood_normal, Loglikelihood_abnormal);
                [MO_real,real_MT,smoothed_B,map_percent_test]=generate_test_seq4d(Or_test, MT_test,T0, N_test,enlarge_final,dict_final,K,ktov,M,All_min_max,smoothed_B_normal,S_min_max,r);
                lkh_normal_test = hsmm_likelihood_2f(A, smoothed_B, P, PI, D, K, M, MO_real, MT_test, N_test,T0,real_MT);
                lkh_normal_test_cal =  round(abs(lkh_normal_test - mean_normal)/std_normal,3);
            
                % save('hsmm_parameter_normal4f_1.mat','r','S_min_max','A_normal', 'B_normal', 'P_normal', 'PI_normal', 'D', 'K', 'M','en','dict_final','MO','MT','N','ktov','All_min_max','Loglikelihood_normal','Loglikelihood_abnormal');
                
                % disp(map_percent_noraml);
                % disp(map_percent_abnormal); 
                % disp(Loglikelihood_normal);
                % disp(Loglikelihood_abnormal); 
                % disp(Loglikelihood_abnormal_cal);
                fprintf('%s\n', mat2str(map_percent_noraml));
                fprintf('%s\n', mat2str(map_percent_abnormal));
                fprintf('%s\n', mat2str(Loglikelihood_normal));
                fprintf('%s\n', mat2str(Loglikelihood_abnormal));
                fprintf('%s\n', mat2str(Loglikelihood_normal_cal));
                fprintf('%s\n', mat2str(Loglikelihood_abnormal_cal));
                disp([map_percent_test,lkh_normal_test,lkh_normal_test_cal]);
                disp([mean_normal,std_normal,variance_normal,variance_abnormal]);
            end
        else
            variance_normal = Inf;
        
        end
    end


    function [dict_final, K, All_min_max, enlarge_final, S_min_max, Orn_final,Orn_back,O] = discrete(en, Or,r)
        %% discrete numeric data to integers
        enlarge_final = en;
        Orn = Or;
        % map to [0,1]
        mino = min(Orn, [], 1);
        Orn = Orn - repmat(mino, size(Orn, 1), 1);
        maxo = max(Orn, [], 1);
        % maxo(maxo == 0) = 0.01;
        All_min_max = [mino; maxo];
        Orn = Orn ./ repmat(maxo, size(Orn, 1), 1);
        Orn(isnan(Orn)) = 0;
        % discrete to [1,enlarge-1]
        % O = floor(sum(Orn, 2));
        O = floor(enlarge_final * sum(Orn .^ r, 2));
        mins = min(O);
        O = O - repmat(mins, size(O, 1), 1);
        maxs = max(O);
        % if maxs == 0
        %     maxs = eps;
        % end
        S_min_max = [mins, maxs];
        O = O ./ repmat(maxs, size(O, 1), 1); % Normalization Method
        O(isnan(O)) = 0;
        % Orn_final = O; % after discreted
        Orn_back = O; % after discreted
        hashMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
        k = 1;
        % Orn_back = O; % after discreted
        Orn_final = O; % after discreted
        for t = 1:size(O, 1)
            keyToFind = O(t); % creat 1

            if isKey(hashMap, keyToFind)
                O(t) = hashMap(keyToFind);
            else
                k = k + 1;
                hashMap(keyToFind) = k;
                O(t) = k;

            end

        end

        dict_final = hashMap;
        K = max(O);
        
    end

    function [ktov, B_init, MO_dist, MO, ktov_or] = init_some_parameter(K, Dim, N, M, O, MT, Or,Orn_final)
        % %map back using average values
        ktov = zeros(K, Dim * N); % k --> obsv
        ktov_or = ktov;
        % ktov_c(ktov_c == 0) = 1;
        % ktov = ktov ./ ktov_c;

        %%% initialize the observation probability matrix B
        B_init = repmat(1 / K, M, K);
        % hasNaN = any(isnan(B_init), 'all');
        % tt = B_init;
        % tt(B_init == 0) = 1;
        % B_init(B_init == 0) = min(tt(:)) / 100.0;

        %%% divide the observation sequence into N sub-sequences
        % MT = ones(1,N) * T0;        % length of each sub-sequence
        MO = zeros(size(Or, 1), 1);
        t_start = 1;
        t_end = 0;

        for on = 1:N % for each observation sequence
            T = MT(on);
            t_end = t_end + T; % the length of the n'th obs seq
            MO([t_start:t_end], :) = Orn_final([t_start:t_end], :); % the n'th observation sequence
            t_start = t_start + T;
        end

        MO_dist = cell(1, N);

        for on = 1:N % for each observation sequence
            T = MT(on);
            % t_end = t_end + T;               % the length of the n'th obs seq
            MO_dist{1, on} = O(1:T); % the n'th observation sequence
            O(1:T) = [];
        end

        MO = MO_dist;
        hasNaN = any(isnan(B_init), 'all');

        if hasNaN
            disp('generate_train_seq2 :Yes');
        end

    end

    
end % not need
