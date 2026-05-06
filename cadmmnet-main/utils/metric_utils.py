import os
import sys
import torch

try:
    from .initialization_utils import initialize_model, load_state
    from .utils import k_largest_peaks_2d
except ImportError:
    from initialization_utils import initialize_model, load_state
    from utils import k_largest_peaks_2d

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def detect_2d(spectrum, ground_truth, N_angle, N_delay, grid_theta, grid_tau, bin_threshold=2, amp_threshold=0.4,
              metric='detection_rate'):
    spec_pk_supp = k_largest_peaks_2d(torch.abs(spectrum), N_angle, N_delay, k=15)
    spec_pk_amp = torch.abs(spectrum[spec_pk_supp]) if len(spec_pk_supp) > 0 else torch.tensor([])
    gt_pk_supp = torch.nonzero(ground_truth).squeeze(-1)
    gt_pk_amp = torch.abs(ground_truth[gt_pk_supp])

    if len(gt_pk_supp) == 0:
        if metric == 'detection_rate':
            return 1.0 if len(spec_pk_supp) == 0 else 0.0
        else:
            return float('nan'), float('nan')

    detections = torch.zeros_like(gt_pk_supp, dtype=torch.float64)
    errors_a = torch.zeros_like(gt_pk_supp, dtype=torch.float64)
    errors_d = torch.zeros_like(gt_pk_supp, dtype=torch.float64)
    assigned = torch.tensor([], dtype=torch.long)

    grid_theta = grid_theta.cpu()
    grid_tau = grid_tau.cpu()

    for k_idx in range(len(gt_pk_supp)):
        gt_a = gt_pk_supp[k_idx] // N_delay
        gt_d = gt_pk_supp[k_idx] % N_delay

        spec_a = spec_pk_supp // N_delay
        spec_d = spec_pk_supp % N_delay

        dist_a = torch.abs(spec_a - gt_a)
        dist_d = torch.abs(spec_d - gt_d)

        detected_indices = torch.where((dist_a <= bin_threshold) & (dist_d <= bin_threshold))[0]

        mask = ~torch.isin(detected_indices, assigned)
        detected_indices = detected_indices[mask]

        if detected_indices.numel() == 0:
            continue

        dist_combined = dist_a[detected_indices] + dist_d[detected_indices]
        sorted_idx = torch.argsort(dist_combined)
        detected_indices = detected_indices[sorted_idx]

        if metric == 'detection_rate':
            if torch.any(spec_pk_amp[detected_indices] > amp_threshold * gt_pk_amp[k_idx]):
                detections[k_idx] = 1

        elif metric == 'rmse':
            for i in detected_indices:
                if spec_pk_amp[i] >= amp_threshold * gt_pk_amp[k_idx]:
                    detections[k_idx] = 1
                    phys_spec_a = grid_theta[spec_a[i]]
                    phys_spec_d = grid_tau[spec_d[i]]
                    phys_gt_a = grid_theta[gt_a]
                    phys_gt_d = grid_tau[gt_d]

                    errors_a[k_idx] = (phys_spec_a - phys_gt_a) ** 2
                    errors_d[k_idx] = (phys_spec_d - phys_gt_d) ** 2
                    break

        if detected_indices.numel() > 0:
            for i in detected_indices:
                if spec_pk_amp[i] >= amp_threshold * gt_pk_amp[k_idx]:
                    assigned = torch.cat((assigned, torch.tensor([i], dtype=torch.long)))
                    break

    if metric == 'detection_rate':
        return torch.sum(detections) / len(detections)
    elif metric == 'rmse':
        if torch.sum(detections) != 0:
            return torch.sum(errors_a) / torch.sum(detections), torch.sum(errors_d) / torch.sum(detections)
        else:
            return float('nan'), float('nan')


def evaluate_model(model, dataset_test_path, num_layers, model_path=None, load_latest_state=False,
                   metric='detection_rate', bin_threshold=2, amp_threshold=0.4, return_degs=True, device='cpu'):
    test_data = torch.load(dataset_test_path, weights_only=True)
    ground_truth = test_data['data']['sparse_vectors'].cpu()
    measurement_vectors = test_data['data']['measurement_vectors'].to(device)
    meta = test_data['metadata']

    # 修复 1: 安全获取 array_type，默认设为 '2d'
    array_type = meta.get('array_type', '2d').lower()

    # --- 获取 SNR 信息 ---
    # 修复 2: 优先从 data 中获取实际的 snr_values，如果失败则回退到 metadata
    data_snr_values = test_data['data'].get('snr_values')
    if data_snr_values is not None:
        snr_values = torch.unique(data_snr_values).tolist()
    else:
        snr_values = meta.get('snr_values', meta.get('snr_range', [0, 5, 10, 15, 20]))

    num_test_vectors = ground_truth.shape[0]

    if 'num_vectors_per_snr' in meta:
        num_vectors_per_snr = meta['num_vectors_per_snr']
    elif 'num_vectors' in meta:
        num_vectors_per_snr = meta['num_vectors'] // len(snr_values)
    else:
        num_vectors_per_snr = num_test_vectors // len(snr_values)

    num_snr_values = len(snr_values)

    # ---------------------------------------
    dictionary_path = meta['dictionary_path']
    dict_data = torch.load(dictionary_path, weights_only=True)

    N_a = len(dict_data['freq_grid_theta'])
    N_d = len(dict_data['freq_grid_tau'])
    grid_theta = dict_data['freq_grid_theta']
    grid_tau = dict_data['freq_grid_tau']

    if model_path or load_latest_state:
        network = initialize_model(model, dictionary_path, num_layers, device)
        model_tag = load_state(network, None, array_type=array_type, load_latest_state=load_latest_state,
                               model_path=model_path, return_tag=True)
        network.eval()
        batch_size = 32
        spectrums_list = []

        print(f"开始使用模型 {model_tag} 进行评估...")

        with torch.no_grad():
            for i in range(0, num_test_vectors, batch_size):
                batch_mv = measurement_vectors[i: i + batch_size]
                batch_spec = network(batch_mv).detach().cpu()
                spectrums_list.append(batch_spec)
        spectrums = torch.cat(spectrums_list, dim=0)
    else:
        raise ValueError("Provide path or set load_latest_state=True")

    if metric == 'rmse':
        results_a = torch.zeros(num_test_vectors, dtype=torch.float64)
        results_d = torch.zeros(num_test_vectors, dtype=torch.float64)
    else:
        results = torch.zeros(num_test_vectors, dtype=torch.float64)

    for s in range(num_test_vectors):
        if metric == 'rmse':
            res_a, res_d = detect_2d(torch.abs(spectrums[s].flatten()), torch.abs(ground_truth[s].flatten()), N_a, N_d,
                                     grid_theta, grid_tau, bin_threshold, amp_threshold, metric)
            results_a[s] = res_a
            results_d[s] = res_d
        elif metric == 'detection_rate':
            res_det = detect_2d(torch.abs(spectrums[s].flatten()), torch.abs(ground_truth[s].flatten()), N_a, N_d,
                                grid_theta, grid_tau, bin_threshold, amp_threshold, metric)
            results[s] = res_det

    if metric == 'nmse':
        results = torch.norm(spectrums.flatten(1), dim=1) ** 2 / (
                torch.norm(ground_truth.flatten(1), dim=1) ** 2 + 1e-12)

    if metric == 'rmse':
        avg_rmse_a = torch.zeros(num_snr_values, dtype=torch.float64)
        avg_rmse_d = torch.zeros(num_snr_values, dtype=torch.float64)

        for n in range(num_snr_values):
            start_idx = int(n * num_vectors_per_snr)
            end_idx = int((n + 1) * num_vectors_per_snr)

            batch_a = results_a[start_idx:end_idx]
            batch_a = batch_a[~torch.isnan(batch_a)]
            avg_rmse_a[n] = torch.sqrt(torch.sum(batch_a) / len(batch_a)) if len(batch_a) > 0 else float('nan')

            batch_d = results_d[start_idx:end_idx]
            batch_d = batch_d[~torch.isnan(batch_d)]
            avg_rmse_d[n] = torch.sqrt(torch.sum(batch_d) / len(batch_d)) if len(batch_d) > 0 else float('nan')

        print(f"\n--- Evaluation Results (Model: {model_tag}) ---")
        print(f"Tested SNR points: {snr_values}")

        metadata_dict = dict_data.get('metadata', {})
        B = metadata_dict.get('bandwidth_hz', None)
        K = metadata_dict.get('num_subcarriers', 64)

        if B is not None:
            df = B / K
            rmse_deg = avg_rmse_a * 114.59
            rmse_ns = (avg_rmse_d / df) * 1e9
            print(f"Average DOA RMSE (Angle in Degrees  ): {rmse_deg.numpy()}")
            print(f"Average TDE RMSE (Delay in ns):        {rmse_ns.numpy()}")
        else:
            print(f"Average DOA RMSE (Normalized Angle): {avg_rmse_a.numpy()}")
            print(f"Average TDE RMSE (Normalized Delay): {avg_rmse_d.numpy()}")
        print("-----------------------------------------------")

    elif metric == 'detection_rate':
        avg_det = torch.zeros(num_snr_values, dtype=torch.float64)
        for n in range(num_snr_values):
            batch = results[int(n * num_vectors_per_snr):int((n + 1) * num_vectors_per_snr)]
            avg_det[n] = torch.sum(batch) / len(batch)
        print(f"Average Detection Rate: {avg_det.numpy()}")

    elif metric == 'nmse':
        avg_nmse = torch.zeros(num_snr_values, dtype=torch.float64)
        for n in range(num_snr_values):
            batch = results[int(n * num_vectors_per_snr):int((n + 1) * num_vectors_per_snr)]
            avg_nmse[n] = torch.mean(batch)
        print(f"Average NMSE (dB): {(10 * torch.log10(avg_nmse)).numpy()}")