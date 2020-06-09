function Y = filterbank(eeg_data, Fs, fb_idx)
[num_chans, ~, no_trials] = size(eeg_data);
Fs=Fs/2;
passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78];
stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72];
Wp = [passband(fb_idx)/Fs, 90/Fs];
Ws = [stopband(fb_idx)/Fs, 100/Fs];
[N, Wn]=cheb1ord(Wp, Ws, 3, 40);
[B, A] = cheby1(N, 0.5, Wn);
Y = zeros(size(eeg_data));
if no_trials == 1
    for i = 1:1:num_chans
        Y(i, :) = filtfilt(B, A, eeg_data(i, :));
    end 
else
    for j = 1:1:no_trials
        for i = 1:1:num_chans
            Y(i, :, j) = filtfilt(B, A, eeg_data(i, :, j));
        end 
    end 
end 
end