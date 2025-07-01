import torch
from torch import nn
from map_pre_old import MapDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks_2 import Denoiser
from utils import plot_trajectories, sample_noise, embed_features
import matplotlib.pyplot as plt
import numpy as np
from infer_2 import calculate_validation_loss_and_plot
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    dataset = MapDataset(
        xml_dir='./real_mixture/cleaneddata/train',
        obs_len=10,
        pred_len=20,
        max_radius=100,
        num_timesteps=30,
        num_polylines=500,
        num_points=10,
        save_plots=False,
        max_agents=32
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)

    model = Denoiser().to(device)  # Move model to device
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0)  # Changed to lr=0 for ramp-up

    # Calculate steps for learning rate ramp-up
    steps_per_epoch = len(dataloader)
    ramp_up_steps = int(0.1 * steps_per_epoch)
    target_lr = 3e-4

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    writer = SummaryWriter(log_dir="./runs_5/REAL")
    num_epochs = 5000
    global_step = 0 
    for i in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move all tensors from the batch to the device
            ego_ids, feature_tensor, feature_mask, roadgraph_tensor, roadgraph_mask, observed, observed_masks, ground_truth, ground_truth_masks, scene_means, scene_stds = batch
            # Clone the tensor (optional if you need the original later)
            feature_tensor_clone = feature_tensor.clone()
            feature_mask_clone = feature_mask.clone()

            # Slice observed part (t=0 to 19)
            obs_part = feature_tensor_clone[:, :, :10, :]
            obs_mask = feature_mask_clone[:, :, :10]

            # Slice future part (t=20, 22, ..., 58)
            future_part = feature_tensor_clone[:, :, 10:, :]
            future_mask = feature_mask_clone[:, :, 10:]

            # Concatenate along the time dimension
            feature_tensor = torch.cat([obs_part, future_part], dim=2).to(device)
            feature_mask = torch.cat([obs_mask, future_mask], dim=2).to(device)

            roadgraph_tensor = roadgraph_tensor.to(device)
            roadgraph_mask = roadgraph_mask.to(device)
            ground_truth = ground_truth.to(device)
            scene_means = scene_means.to(device)
            scene_stds = scene_stds.to(device)

            #if torch.any(torch.all(~feature_mask, dim=-1)):
                #print("Found agents with all timesteps invalid!")
                #all_invalid = torch.all(~feature_mask, dim=-1)
                #for b in range(feature_mask.size(0)):
                    #invalid_agents = torch.where(all_invalid[b])[0]
                    #if invalid_agents.numel() > 0:
                        #print(f"Batch {b}: Agents with all invalid timesteps: {invalid_agents.tolist()}")

            sigma, noised_tensor = sample_noise(feature_tensor)
            c_skip = 0.5**2 / (sigma**2 + (0.5**2))
            c_out = sigma * 0.5 / torch.sqrt((0.5**2) + sigma**2)
            c_in = 1 / torch.sqrt((sigma**2) + 0.5**2)
            c_noise = 1/4 * torch.log(sigma)

            result = noised_tensor.clone()
            result[:, :, 10:, :] = c_in * noised_tensor[:, :, 10:, :]
            embedded = embed_features(result, c_noise)
            model_out = model(embedded, roadgraph_tensor, feature_mask, roadgraph_mask)[:, :, 10:, :]
            #print(c_skip.shape, noised_tensor.shape,c_out.shape, model_out.shape)
            D_theta = c_skip * noised_tensor[:, :, 10:, :] + c_out * model_out

            gt_pred = feature_tensor[:, :, 10:, :]
            mask_pred = feature_mask[:, :, 10:]
            valid_mask = mask_pred.unsqueeze(-1).expand_as(D_theta)

            squared_diff = (D_theta - gt_pred) ** 2
            masked_squared_diff = squared_diff * valid_mask.float()
            loss_per_batch = masked_squared_diff.sum(dim=[1, 2, 3]) / valid_mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)
            #weighted_loss = loss_per_batch.mean()
            # Weighting
            sigma_data = 0.5
            weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
            weighted_loss = (loss_per_batch * weight).mean()

            writer.add_scalar("Loss/Weight", weight[0].item(), global_step)

            optimizer.zero_grad()
            weighted_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            writer.add_scalar("GradNorm", grad_norm.item(), global_step)


            # Set learning rate before optimizer step
            if global_step < ramp_up_steps:
                lr = (global_step / ramp_up_steps) * target_lr
            else:
                lr = target_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


            optimizer.step()
            # Redefine full_valid_gt for plotting
            full_valid_gt = torch.zeros_like(gt_pred)  # Shape: [B, A, T, D]
            full_valid_gt[valid_mask] = gt_pred[valid_mask]  # Use gt_pred directly

            writer.add_scalar("Loss/Iteration", weighted_loss.item(), global_step)
            writer.add_scalar("Sigma", sigma[0].item(), global_step)
            global_step += 1  # Increment global step for each iteration

            epoch_loss += weighted_loss.item()
            num_batches += 1
            #print(f'epoch: {i}, batch: {batch_idx}, loss: {weighted_loss.item()}')

        # Log average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        writer.add_scalar("Loss/Epoch", avg_epoch_loss, i)
        print(f'Epoch {i} completed. Average Loss: {avg_epoch_loss:.4f}')
        if i % 20 == 0:
            torch.save(model.state_dict(), f"model_epoch_{i}_REAL_500_poly_10_pts.pt")
            avg_val_loss, val_fig = calculate_validation_loss_and_plot(
                 model=model,
                 val_xml_dir='./real_mixture/cleaneddata/validation',  # <--- YOUR VALIDATION FOLDER
                 val_batch_size=64,      # Adjust as needed
                 obs_len=10,
                 pred_len=20,
                 max_radius=100,
                 num_polylines=500,
                 num_points=10,
                 max_agents=32,
                 sigma_data=0.5
             )
            print(f"Epoch {i}: Validation Loss = {avg_val_loss:.4f}")

            if not np.isnan(avg_val_loss):
                writer.add_scalar("Loss/Validation", avg_val_loss, i)
            if val_fig:
                writer.add_figure(f"Inference/Validation_Epoch_{i}", val_fig, i)
                plt.close(val_fig) # Close figure to free memory

            #writer.add_figure(f"Inference/Epoch_{i}", fig, i)
            #plt.close(fig)  # Clean up

            for b in range(feature_tensor.size(0)):
                map_polylines = roadgraph_tensor[b]       # Shape: [num_polylines, num_points, 2]
                polyline_masks = roadgraph_mask[b]        # Shape: [num_polylines]
                pred_traj = D_theta[b]                    # Shape: [agents, pred_len, 3]
                gt_traj = full_valid_gt[b]                # Shape: [agents, pred_len, 3]
                noisy_traj = result[b, :, 10:, :]         # Shape: [agents, pred_len, 3]
                trajectory_mask = feature_mask[b, :, 10:] # Shape: [agents, pred_len]
                ego_id = ego_ids[b]

                # Get mean and std for this batch item (assuming they are per-batch)
                mean = scene_means[b]  # Shape: [3] for [x, y, theta]
                std = scene_stds[b]    # Shape: [3] for [x, y, theta]

                # Compute the inverse scaling factor (reverse of 0.5 / std)
                inverse_scale_factor = std / 0.5  # Shape: [3], equivalent to 2 * std

                # Unscale trajectories (x, y, theta)
                pred_traj_unscaled = pred_traj * inverse_scale_factor[None, None, :]  # Shape: [agents, pred_len, 3]
                gt_traj_unscaled = gt_traj * inverse_scale_factor[None, None, :]      # Shape: [agents, pred_len, 3]
                noisy_traj_unscaled = noisy_traj * inverse_scale_factor[None, None, :]  # Shape: [agents, pred_len, 3]

                # Unscale roadgraph polylines (x, y only)
                inverse_scale_xy = inverse_scale_factor[:2]  # Shape: [2] for [x, y]
                map_polylines_unscaled = map_polylines * inverse_scale_xy[None, None, :]  # Shape: [num_polylines, num_points, 2]

                # Generate plot with unscaled data
                fig = plot_trajectories(
                    map_polylines_unscaled, polyline_masks,
                    pred_traj_unscaled, gt_traj_unscaled, noisy_traj_unscaled,
                    trajectory_mask, ego_id, eval=True
                )
                writer.add_figure(f"Trajectories/Batch_{b}_epoch{i}", fig, i)
                plt.close(fig)


writer.close()

