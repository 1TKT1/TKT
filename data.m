%% 5G 高精度定位：CADMM-Net 测试集批量生成器
clear; clc; close all;
% ==========================================
% 0. 随机种子与测试集规模设置
% ==========================================
seed = 999; % 更换种子，确保测试集数据与训练集完全独立
rng(seed); 
snr_values = [0, 5, 10, 15, 20]; 
num_samples_per_snr = 200;       
numSamples = length(snr_values) * num_samples_per_snr; % 总计 1000 个样本
fprintf(' 已设置测试集随机种子: %d\n', seed);
fprintf(' 计划生成 %d 个 SNR 测试点，共计 %d 个样本...\n', length(snr_values), numSamples);

% ==========================================
% 1. 系统参数配置 
% ==========================================
Fs = 122.88e6; Ts = 1/Fs;   
fc = 3.5e9; c = 3e8;        
lambda = c/fc; d = lambda/2;
M = 16;                     
N_fft = 4096;               
num_used_sc_total = 3276;   
comb_size = 8;              
N_cp = 288;                 
M_mod = 16;                 

% ==========================================
% 2. 导频索引与 2D 网格字典初始化
% ==========================================
start_offset = (N_fft - num_used_sc_total) / 2;
comb_indices = round(start_offset + (1 : comb_size : num_used_sc_total));
comb_indices = comb_indices(comb_indices > 0 & comb_indices <= N_fft);
num_active = length(comb_indices);
freq_active = (comb_indices - (N_fft/2 + 1)) * (Fs / N_fft); 
N_theta = 128;                  
N_tau = 128;                    
sin_theta_grid = linspace(sin(deg2rad(-60)), sin(deg2rad(60)), N_theta);
theta_grid = rad2deg(asin(sin_theta_grid));         
tau_grid = linspace(10e-9, 200e-9, N_tau);       
m_idx = (0:M-1).';
A_theta_mat = exp(-1j * 2 * pi * d / lambda * m_idx * sin_theta_grid); 
A_tau_mat = exp(-1j * 2 * pi * freq_active.' * tau_grid);              
A_theta.real = single(real(A_theta_mat)); A_theta.imag = single(imag(A_theta_mat));
A_tau.real = single(real(A_tau_mat));     A_tau.imag = single(imag(A_tau_mat));

% 1. DoA 瑞利分辨率极限: 基于阵元数 M 的正弦空间分辨率 
min_sin_theta_gap = 1 / (M); 
% 2. TDE 带宽分辨率极限: 基于导频信号实际占据的总有效带宽 B 的倒数
B_bandwidth = max(freq_active) - min(freq_active);
min_tau_gap = 1 / B_bandwidth; 

% 初始化存储空间
all_H_cfr_real = zeros(M, num_active, numSamples, 'single');
all_H_cfr_imag = zeros(M, num_active, numSamples, 'single');
all_X_label    = zeros(N_theta, N_tau, numSamples, 'single'); 

% ==========================================
% 3. 批量数据生成主循环
% ==========================================
for i = 1:numSamples
    
    % --- a. 确定当前样本的 SNR ---
    snr_idx = ceil(i / num_samples_per_snr);
    SNR = snr_values(snr_idx);
    
    % --- b. 随机多径参数生成 ---
    L = randi([2, 6]);          
    
    while true
        % 动态产生候选多径参数
        theta_l = -60 + 120 * rand(1, L);       
        tau_l = (10 + 190 * rand(1, L)) * 1e-9; 
        
        is_valid = true;
        % 执行 2D 时空联合排查
        for p1 = 1:L
            for p2 = p1+1:L
                % 计算两两路径间的物理空间正弦差值与绝对时延差值
                sin_diff = abs(sin(deg2rad(theta_l(p1))) - sin(deg2rad(theta_l(p2))));
                tau_diff = abs(tau_l(p1) - tau_l(p2));
                

                if sin_diff < min_sin_theta_gap && tau_diff < min_tau_gap
                    is_valid = false; 
                    break;
                end
            end
            if ~is_valid, break; end
        end
        
        if is_valid
            break; 
        end
    end
    
    % 多径参数解耦锁定后，独立产生复衰落增益
    alpha_l = (randn(1, L) + 1j*randn(1, L)) ./ sqrt(2*L); 
    
    % --- c. 生成 2D 稀疏标签 ---
    X_sparse = zeros(N_theta, N_tau);
    for p = 1:L
        [~, idx_theta] = min(abs(sin_theta_grid - sin(deg2rad(theta_l(p)))));
        [~, idx_tau]   = min(abs(tau_grid - tau_l(p)));
        X_sparse(idx_theta, idx_tau) = X_sparse(idx_theta, idx_tau) + abs(alpha_l(p));
    end
    all_X_label(:,:,i) = single(X_sparse);
    
    % --- d. 发送端：OFDM 信号产生 (Tx) ---
    data_bits = randi([0 M_mod-1], num_active, 1);
    x_qam = qammod(data_bits, M_mod, 'UnitAveragePower', true);
    X_freq = zeros(N_fft, 1);
    X_freq(comb_indices) = x_qam;
    x_time = ifft(ifftshift(X_freq)) * sqrt(N_fft); 
    x_tx = [x_time(end-N_cp+1:end); x_time];        
    
    % --- e. 分数时延信道传输 
    N_sinc = 41; 
    filter_delay = (N_sinc-1)/2; % 计算 sinc 滤波器的群延迟
    n_idx = -filter_delay : filter_delay;
    x_rx_multiantenna = zeros(length(x_tx), M);
    
    for m = 1:M
        for l = 1:L
            phase_shift = exp(-1j * 2*pi * d/lambda * (m-1) * sin(deg2rad(theta_l(l))));
            total_delay_samples = tau_l(l) / Ts;
            int_delay = floor(total_delay_samples);
            frac_delay = total_delay_samples - int_delay;
            
            s_kernel = sinc(n_idx - frac_delay) .* hanning(N_sinc)';
            s_kernel = s_kernel / sum(s_kernel); 
            
            % 抵消滤波器的群延迟，防止时延 TDE 溢出搜索网格
            path_sig_raw = filter(s_kernel, 1, [x_tx; zeros(filter_delay, 1)]); 
            path_sig = path_sig_raw(filter_delay + 1 : end); 
            
            % 然后再叠加真实的物理整数时延
            path_sig = [zeros(int_delay, 1); path_sig(1:end-int_delay)]; 
            
            x_rx_multiantenna(:, m) = x_rx_multiantenna(:, m) + alpha_l(l) * phase_shift * path_sig;
        end
    end
   
    % 保证多测试样本间以及同一 SNR 组内的 Noise Floor 绝对平稳严谨
    x_rx_noisy = awgn(x_rx_multiantenna, SNR, 0); 
    
    % --- f. 接收端：信道估计与 CFR 提取 ---
    H_all = zeros(M, num_active);
    for m = 1:M
        x_no_cp = x_rx_noisy(N_cp+1:end, m);
        Y_freq = fftshift(fft(x_no_cp, N_fft)) / sqrt(N_fft);
        H_all(m, :) = Y_freq(comb_indices) ./ x_qam;
    end
    all_H_cfr_real(:,:,i) = real(single(H_all));
    all_H_cfr_imag(:,:,i) = imag(single(H_all));
    
    if mod(i, 100) == 0
        fprintf('   已处理进度: %d/%d (当前 SNR = %d dB)...\n', i, numSamples, SNR); 
    end
end

% ==========================================
% 4. 保存测试集至 .mat 文件
% ==========================================
targetDir = 'D:\soft\pythonProject\cadmmnet-main\datasets';
if ~exist(targetDir, 'dir'), mkdir(targetDir); end
savePath = fullfile(targetDir, 'Test_Dataset_OFDM_CFR_2D.mat');

all_H_cfr.real = all_H_cfr_real;
all_H_cfr.imag = all_H_cfr_imag;

save(savePath, 'all_H_cfr', 'all_X_label', 'A_theta', 'A_tau', ...
     'theta_grid', 'tau_grid', 'freq_active', ... 
     'snr_values', 'num_samples_per_snr', '-v7.3');
fprintf('测试集生成完毕！群延迟已抵消，标准时空碰撞已消除，保存在: %s\n', savePath);