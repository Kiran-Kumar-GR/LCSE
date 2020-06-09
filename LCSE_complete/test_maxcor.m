function [results,R] = test_maxcor(data_test, model, filter_ab, Recon_channel)
a=filter_ab(1); b=filter_ab(2);
fb=1:model.num_fbs;
fb_coefs = fb.^(-a)+b;
results=zeros(1,model.num_targs);
n=length(Recon_channel);
R=zeros(model.num_targs,model.num_targs);
for i = 1:1:model.num_targs
    r=zeros(model.num_fbs,model.num_targs);
    r1=r;r2=r;
    test_tmp = squeeze(data_test(:, :,i));
    for j = 1:1:model.num_fbs
        testdata = filterbank(test_tmp, model.fs, j);
        for k = 1:1:model.num_targs
            traindata =  squeeze(model.trains(k, j, :, :));
            w = squeeze(model.W(j, k, :, :));
            test=testdata'*w; ref=traindata'*w;
            r_tmp1 = corrcoef(test, ref);
            for l= 1:(n)
                r_tm= corrcoef(test(:,l), ref(:,l));
                r_tmp(l)=r_tm(1,2);
            end
            r1(j,k) = max(mean(r_tmp));
            r2(j,k) = max(r_tmp1(1,2));
        end 
    end 
    rho1 = fb_coefs*r1;
    rho2 = fb_coefs*r2;
    if max(rho1)>max(rho2)
        [~, tau] = max((rho1));
        R(:,i)=rho1;
    else
        [~, tau] = max((rho2));
        R(:,i)=rho2;
    end
    results(i) = tau;
end
end