function model = train_maxcor(data_train, fs, num_fbs, n)
n_c=1:max(n);
[num_chans, num_smpls, num_targs,num_trial] = size(data_train);
trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
W = zeros(num_fbs, num_targs, num_chans, (length(n_c)*num_trial));
for i = 1:1:num_targs
    eeg_tmp = squeeze(data_train(:, :,i,:));
    for j = 1:1:num_fbs
        eeg_tmp = filterbank(eeg_tmp, fs, j);
        trains(i,j,:,:) = squeeze(mean(eeg_tmp, 3));
        w_tmp = maxcor(eeg_tmp,n_c);
        W(j, i, :,:) = w_tmp(:,:);
    end 
end 
model = struct('trains', trains, 'W', W,...
    'num_fbs', num_fbs, 'fs', fs, 'num_targs', num_targs);
end

function We = maxcor(data_train,n_c)
[num_chans, num_smpls, num_trials]  = size(data_train);
S = zeros(num_smpls);
We=[];
% computing G
for trial_i = 1:1:num_trials
    x1 = squeeze(data_train(:,:,trial_i));
    x1=reshape(zscore(x1(:),0,1),size(x1,1),size(x1,2));x1=x1';
    [~,Trans] = reduce_dimension(x1',num_chans-1);
    x=x1*Trans;
    rI=1.0e-07*eye(size(x,2));
    g=(x*((((x'*x)+rI)^-1)*x'));
    g = g*g' + g'*g;
    S=S+g;
end 
[G1,~] = eigs(S);
G =G1(:,n_c);
for trial_i = 1:1:num_trials
    x1 = squeeze(data_train(:,:,trial_i));
    x1=reshape(zscore(x1(:),0,1),size(x1,1),size(x1,2));x1=x1';
    Q=real((((x1'*x1)^-1)*x1')*G);
    We=[We Q];
end 
end

function [rTs,Trans] = reduce_dimension(Ts,q)
[U,S,V] = svd(Ts,'econ');
dd = diag(S);
[~,index] = sort(dd,'descend');
ind   = index(1:q);
S2    = S(ind,ind);
Trans = U(:,ind)*(S2)^-1;
rTs   = V(:,ind)';
end
