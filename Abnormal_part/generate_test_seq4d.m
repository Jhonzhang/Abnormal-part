function [MO,real_MT,smoothed_B,map_percent]=generate_test_seq4d(Or_r, MT,T0,N,enlarge_final,dict_final,K,ktov,M,All_min_max,B,S_min_max,r)
    %Or, MT,T0,N,enlarge_final,dict_final,K,ktov,M,All_min_max

    %% discrete numeric data to integers
    % MO = 0;
    % this_sample_mapped_cnt = zeros(size(Or,1),1);
    % Orn_final(end)
    Or_this = Or_r;
    enlarge = enlarge_final;
    hashMap = dict_final;
    % MO = zeros(T0, N);
    MO = cell(1,N);
    % unseen_observation = [];
    unseen_k = [];
    K_r = K;
    new_K_hashmap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    map_percent = [];
    for on=1:N                 % for each observation sequence
        T=MT(on);
        % T = cell2mat(T);
        % disp([T,size(Or_this)]);
        this_sample_mapped_cnt = zeros(T,1);
        cnt_k = 0;
        Orn = Or_this([1:T],:); % 该子序列
        mino = All_min_max(1,:);
        Orn=Orn-repmat(mino,size(Orn,1),1);
        % Orn(30,:)
        maxo = All_min_max(2,:);
        % maxo(maxo==0) = 0.01;
        Orn=Orn ./ repmat(maxo,size(Orn,1),1);
        Orn(isnan(Orn)) = 0;
        % Orn(30,:)
        % discrete to [1,enlarge-1]
        % O=floor(sum(Orn,2));
        O = floor(enlarge*sum(Orn.^r, 2));
        mins = S_min_max(1);
        O = O - repmat(mins, size(O, 1), 1);
        maxs = S_min_max(2);
        % S_min_max = [mins,maxs];
        O = O ./ repmat(maxs, size(O, 1), 1); % Normalization Method
        % S=round(O .* (enlarge-2))+1;
        O(isnan(O)) = 0;
        S = O;
     
        hashMap = dict_final;
        for t =1:size(S,1)
            keyToFind = S(t);% creat 1
            if isKey(hashMap,keyToFind)
                S(t)  = hashMap(keyToFind);
                this_sample_mapped_cnt(t,1) = S(t);        
            else
                cnt_k = cnt_k+1;
                % O(t) = dict_final(keyHash(O(t)));
                if isKey(new_K_hashmap,keyToFind)
                    K_r = new_K_hashmap(keyToFind);
                    S(t) = K_r;
                    unseen_k = [unseen_k,K_r];
                else
                    % unseen_observation = [unseen_observation,keyToFind];
                    K_r = K_r+1;
                    new_K_hashmap(keyToFind) = K_r;
                    S(t) = K_r;
                    unseen_k = [unseen_k,K_r];
                end
            end
        end
        
        array = this_sample_mapped_cnt;
        mapped_elements = array(array ~= 0);
        mapped_elements_view = mapped_elements(mapped_elements~=1);
        percentage = numel(mapped_elements) / numel(S);
        percentage = round(double(vpa(percentage)),3)* 100;
        map_percent = [map_percent,percentage];
        % disp([T, numel(mapped_elements),percentage,cnt_k]);
        Or_this([1:T],:) = [];
        n_T = T;
        % O(1:n_T) = ones(size(O,1),1);
        MO{1,on}=S(1:n_T);      % the n'th observation sequence
        % MO(1:n_T,on)=S(1:n_T);      % the n'th observation sequence
        real_MT(on) = n_T;
        O(1:n_T)=[];
        S(1:n_T)=[];
    end
    % smoothing_param = 0.01;
    smoothing_param = [];
    % disp(map_percent);
    if ~isempty(unseen_k)
        [uniqueElements, ~, elementCounts] = unique(unseen_k);
        % uniqueElements = unique(unseen_k);
        unseen_observation = uniqueElements;
        counts = histcounts(unseen_k,[uniqueElements,uniqueElements(end)+1]);
        min_non_zero = min(B(B~=0));
        for i = 1:numel(uniqueElements)
            this_new_k_smooth_value = min_non_zero * counts(i) / size(Or_r,1);
            % this_new_k_smooth_value
            smoothing_param = [smoothing_param,this_new_k_smooth_value];
            % disp([num2str(uniqueElements),': ',num2str(counts(i))]);
        end
        smoothed_B = B;
        % disp(smoothing_param);
        expend_smoothing_matrix = repmat(smoothing_param,size(B,1),1);
        smoothed_B(:,unseen_observation) = expend_smoothing_matrix;
    else
        smoothed_B = B;
    end

   
        
                
        
        
            
                
        