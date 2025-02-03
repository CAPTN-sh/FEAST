import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_loss(rounds, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, train_losses, label='Train Loss')
    plt.plot(rounds, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_uncertainty(rounds, train_aleatoric, val_aleatoric, train_epistemic, val_epistemic):
    plt.figure(figsize=(10, 5))
    
    # Plot aleatoric uncertainty
    plt.plot(rounds, train_aleatoric, label='Train Aleatoric Uncertainty', color='blue')
    plt.plot(rounds, val_aleatoric, label='Validation Aleatoric Uncertainty', color='blue', linestyle='--')
    
    # Plot epistemic uncertainty
    plt.plot(rounds, train_epistemic, label='Train Epistemic Uncertainty', color='red')
    plt.plot(rounds, val_epistemic, label='Validation Epistemic Uncertainty', color='red', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    plt.title('Aleatoric and Epistemic Uncertainty Over Rounds')
    plt.legend()
    plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne_with_uncertainty(latent_representations, uncertainties, uncertainty_type='Epistemic Uncertainty', threshold=None):
    # Calculate the mean across the sequence length and feature dimensions if needed
    if latent_representations.ndim == 3:
        latent_representations_mean = latent_representations.mean(axis=1)  # Averaging across sequence length or other dimension
    else:
        latent_representations_mean = latent_representations

    # Ensure that latent_representations_mean is 2D
    if latent_representations_mean.ndim == 1:
        latent_representations_mean = latent_representations_mean.reshape(-1, 1)

    # Perform t-SNE on the latent representations
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_representations_mean)

    # Select uncertainty values
    if isinstance(uncertainties, list) and isinstance(uncertainties[0], dict):
        uncertainty_values = [u[f'{uncertainty_type}'] for u in uncertainties]
    else:
        uncertainty_values = uncertainties
        
    #clipped_labels = np.clip(uncertainty_values, 0, 0.12)  
    # Plot t-SNE results with uncertainty as color mapping
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=uncertainty_values, cmap='viridis', s=5, alpha = 1.0)
    if threshold is not None: 
        for i in range(len(uncertainties)):
            if uncertainties[i] > threshold:
                # 
                plt.scatter(tsne_results[i, 0], tsne_results[i, 1], facecolors='none', edgecolors='red', s=10)

    # Colorbar with larger label font size
    cbar = plt.colorbar(scatter)
    #cbar.set_label(f'{uncertainty_type}', fontsize=20)  # Increase font size
    cbar.ax.tick_params(labelsize=18)

    # Set tick parameters for x and y axis
    plt.tick_params(axis='both', which='major', labelsize=22)
    #plt.title(f't-SNE of Latent Representations with {uncertainty_type.capitalize()}')
    #plt.xlabel('Component 1')
    #plt.ylabel('Component 2')
    plt.show()



import numpy as np
import pandas as pd
import seaborn as sns

def visualize_mean_features(inputs, outputs, num_features=20, num_samples=100):

    # 1. Extract the first `num_features` features from inputs and outputs
    inputs_d = inputs[:, :, :num_features]  # Extract first num_features features
    outputs_d = outputs[:, :, :num_features]  # Extract first num_features features

    # 2. Calculate the mean feature values for the first `num_samples` samples
    subset_input_mean = inputs_d[:num_samples].mean(axis=1)  # Shape: (num_samples, num_features)
    subset_re_mean = outputs_d[:num_samples].mean(axis=1)    # Shape: (num_samples, num_features)

    # 3. Convert the mean values to DataFrame for seaborn heatmap
    subset_input_df = pd.DataFrame(subset_input_mean)
    subset_re_df = pd.DataFrame(subset_re_mean)

    # 4. Normalize the color range across both heatmaps
    input_min = subset_input_df.min().min()
    input_max = subset_input_df.max().max()

    # 5. Create heatmaps for input and reconstructed data
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Input heatmap
    cbar_0 = sns.heatmap(subset_input_df, cmap='YlOrRd', vmin=input_min, vmax=input_max, ax=axes[0])
    axes[0].set_title(f'Input', fontsize=30)
    axes[0].set_ylabel('Samples', fontsize=25)
    axes[0].set_xlabel('Features', fontsize=25)
    
    # Y-axis: Tick every 25 samples
    y_ticks = np.arange(0, num_samples, 25)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_ticks, fontsize=25)

    # X-axis: Tick every 5 features
    x_ticks = np.arange(0, num_features, 5)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_ticks, fontsize=25)

    axes[0].tick_params(axis='both', which='major', labelsize=25)
    
    # Increase font size for colorbar
    cbar_0_colorbar = cbar_0.collections[0].colorbar
    #cbar_0_colorbar.ax.set_ylabel('Value', fontsize=16)
    cbar_0_colorbar.ax.tick_params(labelsize=25)

    # Reconstructed output heatmap
    cbar_1 = sns.heatmap(subset_re_df, cmap='YlOrRd', vmin=input_min, vmax=input_max, ax=axes[1])
    axes[1].set_title(f'Reconstruction', fontsize=30)
    axes[1].set_ylabel('Samples', fontsize=25)
    axes[1].set_xlabel('Features', fontsize=25)
    
    # Y-axis: Tick every 25 samples
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_ticks, fontsize=25)

    # X-axis: Tick every 5 features
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_ticks, fontsize=25)

    axes[1].tick_params(axis='both', which='major', labelsize=25)
    
    # Increase font size for colorbar
    cbar_1_colorbar = cbar_1.collections[0].colorbar
    #cbar_1_colorbar.ax.set_ylabel('Value', fontsize=16)
    cbar_1_colorbar.ax.tick_params(labelsize=25)

    # 6. Adjust the layout to prevent label overlap
    plt.tight_layout()
    plt.show()