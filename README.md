# TCADMM-Net 
本仓库开源了基于模型驱动（Model-driven）深度展开（Deep Unfolding）架构的 5G 通感一体化超分辨率参数估计网络 ——TCADMM-Net。该算法旨在从 5G NR 100MHz 系统的信道频率响应（CFR）数据中，跨越瑞利极限，精细化联合估计多径信号的入射角 (DOA)与 时延 (TDE)，为亚米级高精度室内定位与雷达感知提供核心底层技术支撑。
核心架构演进与物理先验

相比于传统的黑盒深度学习或标准的 LISTA/ADMM 网络，本架构引入了以下硬物理流形约束：
1. 免求逆 2D-FFT 算力引擎：利用多轴频域对角化技术，将原 ADMM 内部极其耗时的全矩阵逆运算降维替换为 2D-FFT 与 IFFT，实现细胞核级的超高速级联迭代。
2. 单位圆恒模流形投影：强制将天线导向矢量（Steering Vector）隐式锁死在纯相位复平面单位圆上，拒绝非理想噪点引发的阵列失真，专职通过微调空间相位来对齐泰勒展开离网格误差（Taylor Off-grid Mismatch）。
3. 弹性可学习物理字典：彻底解决 PyTorch 共轭视图在 Adam 优化器 view_as_real 时的反向梯度崩溃问题，物理字典（`A_theta_H` 与 `A_tau_conj`）完全实体化并参与网络联合进化。
4. 物理残差跳跃流：层间引入带有 `tanh` 振幅锚定的可学习跳跃增益 `gamma`，打通深层梯度死结，强力唤醒被随机背景噪声淹没的微弱次强多径路径。
5. 通过二维时空物理字典的克罗内克解耦，彻底消除了高维感知的维数灾难
 

环境依赖配置
评估与训练所需的核心环境依赖如下：
Python >= 3.8
PyTorch>= 2.0 (强烈推荐启用 CUDA GPU 加速)
h5py >= 3.0 (用于读取高性能 MATLAB 生成的 HDF5/.mat 数据集)
scipy >= 1.7 (用于匈牙利二分图最优匹配算法)
numpy >= 1.20
快速安装依赖：
pip install torch h5py scipy numpy

训练命令：python main.py --config config.json train-model
测试命令：python main.py --config config.json evaluate-model 
