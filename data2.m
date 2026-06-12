%% 5G 高精度定位：CADMM-Net 数据集与字典批量生成器
clear; clc; close all;

% ==========================================
% 0. 随机种子与数据集规模设置
% ==========================================
seed = 42; 
rng(seed); % 固定种子，确保实验可重复
numSamples = 2000; % 批量生成的数据量

% 混合信噪比范围设置
snr_min = 0;  % 最低信噪比 
snr_max = 35; % 最高信噪比
fprintf(' 已设置随机种子: %d\n', seed);
fprintf(' 已开启混合信噪比训练模式，SNR 范围: [%d dB, %d dB]\n', snr_min, snr_max);

% ==========================================
% 1. 系统参数配置 
% ==========================================
Fs = 122.88e6; Ts = 1/Fs;   % 采样率与采样间隔
fc = 3.5e9; c = 3e8;        % 载频与光速
lambda = c/fc; d = lambda/2;% 波长与半波长天线间距
M = 16;                     % 接收天线数 (ULA)
N_fft = 4096;               % FFT 长度
num_used_sc_total = 3276;   % 总可用子载波
comb_size = 8;              % Comb-8 结构
N_cp = 288;                 % 循环前缀长度
M_mod = 16;                 % 16-QAM

% ==========================================
% 2. 导频索引与 2D 网格字典初始化 
% ==========================================
start_offset = (N_fft - num_used_sc_total) / 2;
comb_indices = round(start_offset + (1 : comb_size : num_used_sc_total));
comb_indices = comb_indices(comb_indices > 0 & comb_indices <= N_fft);
num_active = length(comb_indices);

% 计算实际的物理频率偏移 (Hz)，用于生成 A_tau 字典
freq_active = (comb_indices - (N_fft/2 + 1)) * (Fs / N_fft); 

% 设定 2D 网格参数
N_theta = 128;                  
N_tau = 128;                    

% 在角度域和时延域均匀划分
sin_theta_grid = linspace(sin(deg2rad(-60)), sin(deg2rad(60)), N_theta);
theta_grid = rad2deg(asin(sin_theta_grid));         
tau_grid = linspace(10e-9, 200e-9, N_tau);       

% 获取字典网格的绝对物理间距
d_sin_theta = sin_theta_grid(2) - sin_theta_grid(1);
d_tau = tau_grid(2) - tau_grid(1);

% 预生成物理字典 
m_idx = (0:M-1).';
A_theta_mat = exp(-1j * 2 * pi * d / lambda * m_idx * sin_theta_grid); % [M, N_theta]
A_tau_mat = exp(-1j * 2 * pi * freq_active.' * tau_grid);              % [K, N_tau]
A_theta.real = single(real(A_theta_mat)); A_theta.imag = single(imag(A_theta_mat));
A_tau.real = single(real(A_tau_mat));     A_tau.imag = single(imag(A_tau_mat));

% 初始化存储空间
all_H_cfr_real = zeros(M, num_active, numSamples, 'single');
all_H_cfr_imag = zeros(M, num_active, numSamples, 'single');
all_X_label    = zeros(N_theta, N_tau, numSamples, 'single'); 
all_snr_values = zeros(numSamples, 1, 'single'); 
fprintf(' 开始批量生成 CADMM-Net 训练数据集 (共 %d 个样本)...\n', numSamples);

% ==========================================
% 3. 批量数据生成主循环
% ==========================================
for i = 1:numSamples
    
    L = randi([2, 6]);          % 随机生成2到6条路径
    
    SNR = snr_min + (snr_max - snr_min) * rand(); 
    all_snr_values(i) = single(SNR);
    
    alpha_l = (randn(1, L) + 1j*randn(1, L)) ./ sqrt(2*L); % 复增益
    
    % 初始化当前样本的物理多径容器与标签谱图
    theta_l = zeros(1, L);
    tau_l = zeros(1, L);
    X_sparse = zeros(N_theta, N_tau);
    
    % 核心修改：受控离网格误差生成引擎 (Off-grid Error Generation)
    for p = 1:L
        % 在网格内部随机选择一个目标基准点索引 (避开最外边缘防止 asin 越界)
        idx_theta = randi([2, N_theta-1]);
        idx_tau   = randi([2, N_tau-1]);
        
        % 核心注入：在离散网格点的「夹缝」中产生连续随机漂移量
        % 范围严格锁死在 [-0.5, 0.5] 个网格间距内
        offset_sin = (rand() - 0.5) * d_sin_theta;
        offset_tau = (rand() - 0.5) * d_tau;
        
        % 合成真实的、带有离网格偏差的连续域物理多径参数
        sin_theta_off = sin_theta_grid(idx_theta) + offset_sin;
        theta_l(p) = rad2deg(asin(sin_theta_off)); % 连续真实角度
        tau_l(p) = tau_grid(idx_tau) + offset_tau;     % 连续真实时延
        
        % 写入2D稀疏标签：由于漂移量严格在半个网格内，其最近邻对齐网格必然是 idx
        X_sparse(idx_theta, idx_tau) = X_sparse(idx_theta, idx_tau) + abs(alpha_l(p));
    end
    all_X_label(:,:,i) = single(X_sparse);
    
    % --- c. 发送端：OFDM 信号产生 (Tx) ---
    data_bits = randi([0 M_mod-1], num_active, 1);
    x_qam = qammod(data_bits, M_mod, 'UnitAveragePower', true);
    X_freq = zeros(N_fft, 1);
    X_freq(comb_indices) = x_qam;
    x_time = ifft(ifftshift(X_freq)) * sqrt(N_fft); 
    x_tx = [x_time(end-N_cp+1:end); x_time];        
    
    % --- d. 分数时延信道传输---
    N_sinc = 41; 
    filter_delay = (N_sinc-1)/2; 
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
            
            path_sig_raw = filter(s_kernel, 1, [x_tx; zeros(filter_delay, 1)]); 
            path_sig = path_sig_raw(filter_delay + 1 : end); 
            
            path_sig = [zeros(int_delay, 1); path_sig(1:end-int_delay)]; 
            
            x_rx_multiantenna(:, m) = x_rx_multiantenna(:, m) + alpha_l(l) * phase_shift * path_sig;
        end
    end
    
    % 叠加噪声
    x_rx_noisy = awgn(x_rx_multiantenna, SNR, 0); 
    
    % --- e. 接收端：信道估计与 CFR 提取 ---
    H_all = zeros(M, num_active);
    for m = 1:M
        x_no_cp = x_rx_noisy(N_cp+1:end, m);
        Y_freq = fftshift(fft(x_no_cp, N_fft)) / sqrt(N_fft);
        H_all(m, :) = Y_freq(comb_indices) ./ x_qam;
    end
    
    all_H_cfr_real(:,:,i) = real(single(H_all));
    all_H_cfr_imag(:,:,i) = imag(single(H_all));
    
    % ==========================================
    % 4. 结果可视化 
    % ==========================================
    if i == 1
        figure('Color', 'w', 'Position', [100, 100, 1000, 800]);
        subplot(2,2,1);
        t_ns = (0:length(x_tx)-1) * Ts * 1e9;
        plot(t_ns, abs(x_tx), 'k', 'LineWidth', 1); hold on;
        plot(t_ns, abs(x_rx_noisy(:, 1)), 'r');
        title(sprintf('1. 时域信号对比 (SNR = %.1f dB)', SNR));
        xlabel('时间 (ns)'); ylabel('幅度'); grid on; xlim([0 1500]);
        
        subplot(2,2,2);
        f_axis = linspace(-Fs/2, Fs/2, num_active) / 1e6;
        plot(f_axis, 20*log10(abs(H_all(1, :))), 'LineWidth', 1.5);
        title('2. 提取的 CFR 幅度响应 (反映多径干涉)');
        xlabel('频率偏移 (MHz)'); ylabel('幅度 (dB)'); grid on;
        
        subplot(2,2,3);
        imagesc(angle(H_all)); colorbar; colormap hsv;
        title('3. 2D 相位指纹 (Angle of H)');
        xlabel('子载波索引'); ylabel('天线索引');
        
        subplot(2,2,4);
        theta_range = -90:0.5:90;
        a_test = exp(-1j * 2*pi * d/lambda * (0:M-1)' * sin(deg2rad(theta_range)));
        spatial_spectrum = abs(a_test' * H_all(:, floor(num_active/2)));
        plot(theta_range, spatial_spectrum/max(spatial_spectrum), 'LineWidth', 1.5);
        hold on; stem(theta_l, ones(size(theta_l)), 'r--');
        title('4. 空间谱切片 (带有离网格误差的真实多径位置)');
        xlabel('角度 (deg)'); legend('估计谱', '离网多径真值'); grid on;
        drawnow; 
    end
    
    if mod(i, 200) == 0, fprintf('   已处理进度: %d/%d...\n', i, numSamples); end
end

% ==========================================
% 5. 保存数据集至 .mat 文件
% ==========================================
targetDir = 'D:\soft\pythonProject\cadmmnet-main\datasets';
if ~exist(targetDir, 'dir'), mkdir(targetDir); end
savePath = fullfile(targetDir, 'Train_Dataset_OFDM_CFR_2D_with_Dict.mat');

all_H_cfr.real = all_H_cfr_real;
all_H_cfr.imag = all_H_cfr_imag;

save(savePath, 'all_H_cfr', 'all_X_label', 'A_theta', 'A_tau', ...
     'theta_grid', 'tau_grid', 'freq_active', 'all_snr_values', '-v7.3');
fprintf('训练数据集生成完毕！保存在: %s\n', savePath);