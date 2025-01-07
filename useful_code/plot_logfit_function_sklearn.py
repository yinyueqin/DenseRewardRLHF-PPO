import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

def load_data(input_file):
    with open(input_file, 'r') as f:
        loaded_data = json.load(f)
    
    positions = sorted(loaded_data['mean'].keys(), key=float)
    positions_float = np.array([float(pos) for pos in positions])
    means = np.array([loaded_data['mean'][pos] for pos in positions])
    stds = np.array([loaded_data['std'][pos] for pos in positions])
    
    return positions_float, means, stds

def fit_huber_log(X, y, epsilon=1.35):
    X_log = np.log(X.reshape(-1, 1) + 1e-10)
    
    huber = HuberRegressor(epsilon=epsilon)
    huber.fit(X_log, y)
    
    return huber.coef_[0], huber.intercept_

def plot_fits(positions, means, stds, mean_params, std_params, save_path):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_xlabel('Segment Position (Percentage)')
    ax1.set_ylabel('Mean Reward', color='tab:blue')
    ax1.scatter(positions, means, color='tab:blue', label='Mean Data', alpha=0.5)
    
    x_smooth = np.linspace(min(positions), max(positions), 1000)
    y_mean_fit = mean_params[0] * np.log(x_smooth + 1e-10) + mean_params[1]
    ax1.plot(x_smooth, y_mean_fit, color='red', 
             label=f'Mean Huber Log Fit: y = {mean_params[0]:.4f}*log(x) + {mean_params[1]:.4f}')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Standard Deviation of Reward', color='tab:orange')
    ax2.scatter(positions, stds, color='tab:orange', label='Std Data', alpha=0.5)
    
    y_std_fit = std_params[0] * np.log(x_smooth + 1e-10) + std_params[1]
    ax2.plot(x_smooth, y_std_fit, color='purple',
             label=f'Std Huber Log Fit: y = {std_params[0]:.4f}*log(x) + {std_params[1]:.4f}')
    
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title('Mean and Standard Deviation of Reward vs Segment Position with Huber Log Fits')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    input_file = 'phi3-instruct_preference700k_1.75_avg_0.5_peak_segment_position_rewards_samples60000.json'
    save_dir = './saved_logfit_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    saved_dir = os.path.join(save_dir, file_name)
    
    positions, means, stds = load_data(input_file)
    
    mean_params = fit_huber_log(positions, means)
    std_params = fit_huber_log(positions, stds)
    
    save_path = f'{saved_dir}_combined_plot_with_huber_log_fits.png'
    plot_fits(positions, means, stds, mean_params, std_params, save_path)
    
    mean_pred = mean_params[0] * np.log(positions + 1e-10) + mean_params[1]
    std_pred = std_params[0] * np.log(positions + 1e-10) + std_params[1]
    
    r2_mean = calculate_r2(means, mean_pred)
    r2_std = calculate_r2(stds, std_pred)
    
    print(f"Mean Huber Log Fit: y = {mean_params[0]:.4f}*log(x) + {mean_params[1]:.4f}")
    print(f"Mean R² value: {r2_mean:.4f}")
    print(f"Std Huber Log Fit: y = {std_params[0]:.4f}*log(x) + {std_params[1]:.4f}")
    print(f"Std R² value: {r2_std:.4f}")
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    main()