import torch
import matplotlib.pyplot as plt
import numpy as np
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(map_polylines, polyline_masks, pred_traj, gt_traj, noisy_traj, trajectory_mask, ego_id, save_path=None, eval=False, title_suffix=""):
    """
    Plot predicted and ground truth trajectories side by side in a single figure for all agents.
    Handles both PyTorch Tensors and NumPy arrays as input.

    Args:
        map_polylines (torch.Tensor or np.ndarray): Map polylines [P, N, 2] (x, y coordinates).
        polyline_masks (torch.Tensor or np.ndarray): Mask for valid polylines [P].
        pred_traj (torch.Tensor or np.ndarray): Predicted trajectories [A, T_future, 3] (x, y, theta).
        gt_traj (torch.Tensor or np.ndarray): Ground truth trajectories [A, T_future, 3] (x, y, theta).
        noisy_traj (torch.Tensor or np.ndarray): Noisy trajectories [A, T_future, 3] (can be used for viz).
        trajectory_mask (torch.Tensor or np.ndarray): Mask for valid trajectory timesteps [A, T_future].
        ego_id (int or str): Ego agent ID.
        save_path (str, optional): Path to save figure (without extension).
        eval (bool): If False and save_path is None, displays plot. If True, returns figure.
        title_suffix (str, optional): Additional text for the figure title.

    Returns:
        matplotlib.figure.Figure or None: Matplotlib figure object if eval=True, otherwise None.
    """
    # --- Convert inputs to NumPy if they are tensors ---
    if isinstance(map_polylines, torch.Tensor):
        map_polylines = map_polylines.detach().cpu().numpy()
    if isinstance(polyline_masks, torch.Tensor):
        polyline_masks = polyline_masks.detach().cpu().numpy()
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.detach().cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.detach().cpu().numpy()
    if isinstance(noisy_traj, torch.Tensor): # Added conversion for noisy_traj
        noisy_traj = noisy_traj.detach().cpu().numpy()
    if isinstance(trajectory_mask, torch.Tensor):
        trajectory_mask = trajectory_mask.detach().cpu().numpy()

    # --- Input Validation ---
    # Check for NaN/Inf and print warnings
    nan_inf_found = False
    for name, arr in [("map_polylines", map_polylines), ("pred_traj", pred_traj), ("gt_traj", gt_traj), ("noisy_traj", noisy_traj)]:
        if np.any(np.isnan(arr)):
            print(f"Warning: {name} contains NaN values")
            nan_inf_found = True
        if np.any(np.isinf(arr)):
            print(f"Warning: {name} contains infinite values")
            nan_inf_found = True
    # Optional: If NaN/Inf found, return None or an empty figure?
    # if nan_inf_found: return None

    A = pred_traj.shape[0]  # Number of agents

    # --- Compute plot limits based on map and *valid* trajectories ---
    valid_polylines_list = [map_polylines[i] for i in range(map_polylines.shape[0]) if polyline_masks[i]]
    all_x_road = np.concatenate([p[:, 0] for p in valid_polylines_list if p.shape[0] > 0]) if valid_polylines_list else np.array([])
    all_y_road = np.concatenate([p[:, 1] for p in valid_polylines_list if p.shape[0] > 0]) if valid_polylines_list else np.array([])

    all_x_traj, all_y_traj = [], []
    for a in range(A):
        # Use boolean mask directly for indexing
        valid_mask_a = trajectory_mask[a].astype(bool)
        if np.any(valid_mask_a):
            all_x_traj.append(pred_traj[a, valid_mask_a, 0])
            all_y_traj.append(pred_traj[a, valid_mask_a, 1])
            all_x_traj.append(gt_traj[a, valid_mask_a, 0])
            all_y_traj.append(gt_traj[a, valid_mask_a, 1])

    # Concatenate only if there's data
    all_x_traj_np = np.concatenate(all_x_traj) if all_x_traj else np.array([])
    all_y_traj_np = np.concatenate(all_y_traj) if all_y_traj else np.array([])

    all_x = np.concatenate([all_x_road, all_x_traj_np])
    all_y = np.concatenate([all_y_road, all_y_traj_np])

    # Determine limits, handle empty data case
    if all_x.size > 0 and all_y.size > 0:
        # Filter out potential NaN/Inf before calculating min/max
        finite_x = all_x[np.isfinite(all_x)]
        finite_y = all_y[np.isfinite(all_y)]
        if finite_x.size > 0 and finite_y.size > 0:
            xmin, xmax = np.min(finite_x), np.max(finite_x)
            ymin, ymax = np.min(finite_y), np.max(finite_y)
            x_range = max(xmax - xmin, 1.0) # Min range of 1.0
            y_range = max(ymax - ymin, 1.0)
            padding = max(x_range, y_range) * 0.1 # 10% padding based on max range
            xlim = (xmin - padding, xmax + padding)
            ylim = (ymin - padding, ymax + padding)
        else:
            print("Warning: No finite data points found for limit calculation. Using default limits.")
            xlim = (-20, 20)
            ylim = (-20, 20)
    else:
        print("Warning: No map or trajectory data to determine plot limits. Using default limits.")
        xlim = (-20, 20)
        ylim = (-20, 20)

    # --- Plotting ---
    fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True) # Share both axes

    # Plot map polylines on both axes
    for ax in [ax_pred, ax_gt]:
        for i in range(map_polylines.shape[0]):
            if polyline_masks[i]:
                poly = map_polylines[i]
                # Ensure polyline is not just padding zeros before plotting
                if poly.shape[0] > 0 and not np.allclose(poly, 0.0):
                     ax.plot(poly[:, 0], poly[:, 1], color='gray', alpha=0.5, linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X (meters)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal

    ax_pred.set_ylabel('Y (meters)') # Label only on the left

    cmap = plt.get_cmap('hsv', A if A > 1 else 2)

    # Plot predicted trajectories (left subplot)
    for a in range(A):
        valid_mask_a = trajectory_mask[a].astype(bool)
        if np.any(valid_mask_a):
            x = pred_traj[a, valid_mask_a, 0]
            y = pred_traj[a, valid_mask_a, 1]
            theta = pred_traj[a, valid_mask_a, 2]
            ax_pred.plot(x, y, color=cmap(a % cmap.N), linewidth=1.5, alpha=0.9) # Use modulo for safety

            # Add arrow at the start of the valid prediction
            x_start, y_start, theta_start = x[0], y[0], theta[0]
            arrow_length = 2.0
            ax_pred.arrow(
                x_start, y_start,
                arrow_length * np.cos(theta_start), arrow_length * np.sin(theta_start),
                color=cmap(a % cmap.N), head_width=0.5, head_length=1.0, alpha=0.9, length_includes_head=True
            )

    # Plot ground truth trajectories (right subplot)
    for a in range(A):
        valid_mask_a = trajectory_mask[a].astype(bool)
        if np.any(valid_mask_a):
            x = gt_traj[a, valid_mask_a, 0]
            y = gt_traj[a, valid_mask_a, 1]
            theta = gt_traj[a, valid_mask_a, 2]
            ax_gt.plot(x, y, color=cmap(a % cmap.N), linewidth=1.5, alpha=0.9) # Use modulo

            # Add arrow at the start of the valid ground truth
            x_start, y_start, theta_start = x[0], y[0], theta[0]
            arrow_length = 2.0
            ax_gt.arrow(
                x_start, y_start,
                arrow_length * np.cos(theta_start), arrow_length * np.sin(theta_start),
                color=cmap(a % cmap.N), head_width=0.5, head_length=1.0, alpha=0.9, length_includes_head=True
            )

    ax_pred.set_title('Predicted Trajectories')
    ax_gt.set_title('Ground Truth Trajectories')

    sup_title = f'Future Trajectories (Ego ID: {ego_id})'
    if title_suffix: sup_title += f" - {title_suffix}"
    fig.suptitle(sup_title, fontsize=14, y=0.98) # Adjust y slightly down

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout rect to make space for suptitle

    # --- Handle Output ---
    if eval:
        return fig # Return figure object for TensorBoard etc.
    elif save_path:
        try:
            # Suppress UserWarning about tight_layout caused by aspect ratio
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning)
                 fig.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}.png")
        except Exception as e:
            print(f"Failed to save plot {save_path}.png: {e}")
        plt.close(fig) # Close figure after saving
        return None
    else:
         # Display plot if not eval and no save path
         plt.show()
         plt.close(fig)
         return None

def sample_noise(inputs):
    B, A, T, F = inputs.size()
    T_future = 20
    ln_sigma = torch.normal(mean=-1.2, std=1.2, size=(B,), device=inputs.device)
    sigma = torch.exp(ln_sigma).view(B, 1, 1, 1)
    epsilon = torch.randn(B, A, T_future, F, device=inputs.device)
    noised_inputs = inputs.clone()
    noised_inputs[:, :, 10:, :] = noised_inputs[:, :, 10:, :] + epsilon * sigma

    return sigma, noised_inputs

def embed_features(inputs, sigma, embedding_dim=256, T_obs=10, eval=False):
    """
    Embed features using sinusoidal positional encodings for diffusion time tau, scenario time t,
    and agent states x, y, theta.
    
    Args:
        inputs (torch.Tensor): Input tensor [B, A, T, F], where F=3 for x, y, theta.
                              Contains observed states (xobs) for t<T_obs and noisy states (xlat,τ) for t>=T_obs.
        sigma (torch.Tensor): Noise levels [B,], one per batch.
        embedding_dim (int): Dimension of each encoding vector, default 256.
        T_obs (int): Number of observed time steps (default: 20).
        eval (bool): Evaluation mode flag.
    
    Returns:
        torch.Tensor: Embedded features [B, A, T, 5 * embedding_dim]
    """
    # Extract dimensions
    B, A, T, F = inputs.size()
    assert F == 3, "Expected F=3 for x, y, theta"
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    device = inputs.device

    # Generate scenario time t as integers [0, T-1]
    t = torch.arange(0, T, dtype=torch.float32, device=device)  # [T,]

    # Helper function for sinusoidal encoding
    def sinusoidal_encoding(values, min_period, max_period):
        num_freqs = embedding_dim // 2  # e.g., 128 for embedding_dim=256
        i = torch.arange(num_freqs, device=device, dtype=torch.float32)
        exp_term = i / (num_freqs - 1) if num_freqs > 1 else torch.zeros_like(i)
        wavelengths = min_period * (max_period / min_period) ** exp_term
        angular_freqs = 2 * torch.pi / wavelengths
        phases = values[..., None] * angular_freqs
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        encoding = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)  # [..., embedding_dim]
        return encoding

    # 1. Encode agent state features (x, y, theta)
    state_encodings = []
    for f in range(F):  # F=3
        feat = inputs[:, :, :, f]  # [B, A, T]
        enc = sinusoidal_encoding(feat, min_period=0.01, max_period=10)  # [B, A, T, embedding_dim]
        state_encodings.append(enc)

    # 2. Encode scenario time t
    t_enc = sinusoidal_encoding(t, min_period=1, max_period=100)  # [T, embedding_dim]
    t_enc = t_enc.expand(B, A, T, embedding_dim)  # [B, A, T, embedding_dim]

    # 3. Encode diffusion time tau
    if not eval:
        sigma = sigma.squeeze()  # [B,]
    # Create tau tensor: 0 for t < T_obs, sigma for t >= T_obs
    latent_mask = (t >= T_obs).float()  # [T,], 0 for observed, 1 for latent
    tau = sigma[:, None, None] * latent_mask[None, None, :]  # [B, 1, T]
    tau = tau.expand(B, A, T)  # [B, A, T], broadcast across agents
    tau_enc = sinusoidal_encoding(tau, min_period=0.1, max_period=10000)  # [B, A, T, embedding_dim]

    # Concatenate all encodings: x, y, theta, t, tau
    all_encodings = state_encodings + [t_enc, tau_enc]  # 5 tensors, each [B, A, T, embedding_dim]
    embedded = torch.cat(all_encodings, dim=-1)  # [B, A, T, 5 * embedding_dim], e.g., [B, A, T, 1280]
    
    return embedded

if __name__ == "__main__":
    print("all good!")