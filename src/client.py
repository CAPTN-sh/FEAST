import torch
import numpy as np
from flwr.client import NumPyClient
import torch.nn.functional as F

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_uncertainties(v, alpha, beta):
    epsilon = 1e-6
    aleatoric_uncertainty = beta / (alpha - 1 + epsilon)
    epistemic_uncertainty = beta / (v * (alpha - 1 + epsilon))
    return aleatoric_uncertainty, epistemic_uncertainty

def calculate_percentile_thresholds(uncertainties, epistemic_percentile=95, aleatoric_percentile=50):
    """
    Thresholds for epistemic and aleatoric uncertainties were calculated based on the quantile.
    """
    epistemic_threshold = np.percentile([u['epistemic_uncertainty'] for u in uncertainties], epistemic_percentile)
    aleatoric_threshold = np.percentile([u['aleatoric_uncertainty'] for u in uncertainties], aleatoric_percentile)
    return epistemic_threshold, aleatoric_threshold

def train(model, criterion, optimizer, train_dataloader, lambda_reg, offset, device):
    model.train()
    total_train_loss = 0.0
    total_aleatoric_uncertainty = 0.0
    total_epistemic_uncertainty = 0.0
    total_samples = 0
    per_sample_uncertainties = []  # Used to store uncertainty for each sample

    for batch_idx, batch in enumerate(train_dataloader):
        features = batch['inputs'].to(device)
        masks = batch.get('input_masks', None)
        if masks is not None:
            masks = masks.to(device)
            if masks.dim() == 2:
                masks = masks.unsqueeze(-1).expand_as(features)
            # Expand mask dimensions to match the shape of the outputs
            # masks = masks.unsqueeze(-1).expand_as(features).float()
        optimizer.zero_grad()
        mu, v, alpha, beta = model(features, padding_mask=masks)
        
        # per_sample_reconstruction_error = masked_reconstruction_loss(features, mu, masks[:,:,0], batch_mean=False)
        
        loss = criterion((mu, v, alpha, beta), features, lambda_reg, offset, mask=masks, recon_error=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = features.size(0)
        total_train_loss += loss.item() * batch_size

        # Calculation uncertainty
        aleatoric_uncertainty, epistemic_uncertainty = calculate_uncertainties(v, alpha, beta)
        # Average seq_len and feature_dim to get the uncertainty for each sample
        per_sample_aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=[1, 2])  # Shape: [batch_size]
        per_sample_epistemic_uncertainty = epistemic_uncertainty.mean(dim=[1, 2])   # Shape: [batch_size]

        # Cumulative uncertainty for all samples
        total_aleatoric_uncertainty += per_sample_aleatoric_uncertainty.sum().item()
        total_epistemic_uncertainty += per_sample_epistemic_uncertainty.sum().item()
        total_samples += batch_size

        # Store the uncertainty of each sample
        for i in range(batch_size):
            per_sample_uncertainties.append({
                'sample_idx': total_samples + i,
                'aleatoric_uncertainty': per_sample_aleatoric_uncertainty[i].item(),
                'epistemic_uncertainty': per_sample_epistemic_uncertainty[i].item()
            })

        # total_samples += batch_size

    avg_train_loss = total_train_loss / total_samples
    avg_aleatoric_uncertainty_sample = total_aleatoric_uncertainty / total_samples
    avg_epistemic_uncertainty_sample = total_epistemic_uncertainty / total_samples


    return avg_train_loss, per_sample_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample

def masked_reconstruction_loss(original, reconstructed, mask, batch_mean=True):

    error = F.mse_loss(reconstructed, original, reduction="none")  # (batch_size, seq_len, feature_dim)
    error = error.mean(dim=-1)  #  (batch_size, seq_len)
    #print(error.shape, mask.shape)
    masked_error = error * mask  # 

    loss = masked_error.sum(dim=1) / mask.sum(dim=1) 
    if batch_mean == True:
        loss = loss.mean()  
    else:
        pass
    return loss	
    
def evaluate_local(model, criterion, val_dataloader, lambda_reg, offset, device, return_latent=False):
    model.eval()
    total_val_loss = 0.0
    total_aleatoric_uncertainty = 0.0
    total_epistemic_uncertainty = 0.0
    total_samples = 0
    per_sample_uncertainties = []  # Used to store uncertainty for each sample
    latent_representations = []    # Used to store latent representations
    recon_error = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            features = batch['inputs'].to(device)
            masks = batch.get('input_masks', None)
            if masks is not None:
                masks = masks.to(device)
                if masks.dim() == 2:
                    masks = masks.unsqueeze(-1).expand_as(features)
            if return_latent:
                mu, v, alpha, beta, latent = model(features, return_latent=True)
                latent_representations.append(latent.cpu())
            else:
                mu, v, alpha, beta = model(features)
            
            per_sample_reconstruction_error = masked_reconstruction_loss(features, mu, masks[:,:,0], batch_mean=False)
            recon_error.extend(per_sample_reconstruction_error.cpu().numpy())
            
            val_loss = criterion((mu, v, alpha, beta), features, lambda_reg, offset, mask=masks, recon_error=per_sample_reconstruction_error)
            batch_size = features.size(0)
            total_val_loss += val_loss.item() * batch_size

            # Calculation uncertainty
            aleatoric_uncertainty, epistemic_uncertainty = calculate_uncertainties(v, alpha, beta)
            # Average seq_len and feature_dim to get the uncertainty for each sample
            per_sample_aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=[1, 2])  # Shape: [batch_size]
            per_sample_epistemic_uncertainty = epistemic_uncertainty.mean(dim=[1, 2])   # Shape: [batch_size]
            # per_sample_reconstruction_error = masked_reconstruction_loss(features, mu, masks[:,:,0], batch_mean=False)
            # recon_error.extend(per_sample_reconstruction_error.cpu().numpy())
            # Cumulative uncertainty for all samples
            total_aleatoric_uncertainty += per_sample_aleatoric_uncertainty.sum().item()
            total_epistemic_uncertainty += per_sample_epistemic_uncertainty.sum().item()
            total_samples += batch_size

            # Store the uncertainty of each sample
            for i in range(batch_size):
                per_sample_uncertainties.append({
                    'sample_idx': total_samples + i,
                    'aleatoric_uncertainty': per_sample_aleatoric_uncertainty[i].item(),
                    'epistemic_uncertainty': per_sample_epistemic_uncertainty[i].item()
                })

            total_samples += batch_size

    avg_val_loss = total_val_loss / total_samples
    avg_aleatoric_uncertainty_sample = total_aleatoric_uncertainty / total_samples
    avg_epistemic_uncertainty_sample = total_epistemic_uncertainty / total_samples


    if return_latent:
        # 
        latent_representations = torch.cat(latent_representations, dim=0)
        return avg_val_loss, per_sample_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, latent_representations, recon_error
    else:
        return avg_val_loss, per_sample_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, recon_error

def train_and_evaluate_local(model, criterion, optimizer, train_dataloader, val_dataloader,
                             num_epochs=10, lambda_reg=0.01, offset=2, device='cuda', return_latent=False,
                             save_model_path=None):
    model.to(device)
    
    train_losses = []
    val_losses = []
    train_aleatoric_uncertainties = []
    train_epistemic_uncertainties = []
    val_aleatoric_uncertainties = []
    val_epistemic_uncertainties = []
    
    train_aleatoric_uncertainties_avg = []
    train_epistemic_uncertainties_avg = []
    val_aleatoric_uncertainties_avg = []
    val_epistemic_uncertainties_avg = []
    latent_representations = None  
    
    for epoch in range(num_epochs):
        # training
        avg_train_loss, train_uncertainties, avg_aleatoric_train, avg_epistemic_train = train(
            model, criterion, optimizer, train_dataloader, lambda_reg, offset, device
        )

        # validation
        if return_latent:
            results = evaluate_local(
                model, criterion, val_dataloader, lambda_reg, offset, device, return_latent=True
            )
            avg_val_loss, val_uncertainties, avg_aleatoric_val, avg_epistemic_val, latent_reps, recon_error = results
            latent_representations = latent_reps.cpu().numpy()  # 
        else:
            avg_val_loss, val_uncertainties, avg_aleatoric_val, avg_epistemic_val, recon_error = evaluate_local(
                model, criterion, val_dataloader, lambda_reg, offset, device, return_latent=False
            )
        
        current_lr = get_current_lr(optimizer)
        # 
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_aleatoric_uncertainties_avg.append(avg_aleatoric_train)
        train_epistemic_uncertainties_avg.append(avg_epistemic_train)
        val_aleatoric_uncertainties_avg.append(avg_aleatoric_val)
        val_epistemic_uncertainties_avg.append(avg_epistemic_val)
        
        if epoch == num_epochs - 1:
            train_aleatoric_uncertainties.extend([u['aleatoric_uncertainty'] for u in train_uncertainties])
            train_epistemic_uncertainties.extend([u['epistemic_uncertainty'] for u in train_uncertainties])
            val_aleatoric_uncertainties.extend([u['aleatoric_uncertainty'] for u in val_uncertainties])
            val_epistemic_uncertainties.extend([u['epistemic_uncertainty'] for u in val_uncertainties])

        # Anomaly detection with epistemic uncertainty
        epistemic_threshold, _ = calculate_percentile_thresholds(val_uncertainties, epistemic_percentile=95)
        anomalies_epistemic = [u for u in val_uncertainties if u['epistemic_uncertainty'] > epistemic_threshold]

        # Anomaly detection combining aleatoric and epistemic uncertainties
        epistemic_threshold_combined, aleatoric_threshold_combined = calculate_percentile_thresholds(
            val_uncertainties, epistemic_percentile=95, aleatoric_percentile=50
        )
        anomalies_combined = [
            u for u in val_uncertainties 
            if u['epistemic_uncertainty'] > epistemic_threshold_combined and u['aleatoric_uncertainty'] < aleatoric_threshold_combined
        ]
         # Training and evaluation results
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"{'Train Loss:':<30} {avg_train_loss:<10.4f}")
        print(f"{'Train Aleatoric Uncertainty (Avg):':<30} {avg_aleatoric_train:<10.4f}")
        print(f"{'Train Epistemic Uncertainty (Avg):':<30} {avg_epistemic_train:<10.4f}")
        
        print(f"{'Eval Loss:':<30} {avg_val_loss:<10.4f}")
        print(f"{'Eval Aleatoric Uncertainty (Avg):':<30} {avg_aleatoric_val:<10.4f}")
        print(f"{'Eval Epistemic Uncertainty (Avg):':<30} {avg_epistemic_val:<10.4f}")
        print(f"{'Current Learning Rate:':<30} {current_lr:<10.6f}")
        
        print(f"{'Epistemic Percentile Threshold (95th):':<30} {epistemic_threshold:<10.4f}")
        print(f"{'Detected anomalies (epistemic):':<30} {len(anomalies_epistemic)} / {len(val_dataloader.dataset)}")
        
        print(f"{'Combined Threshold - Epistemic (95th):':<30} {epistemic_threshold_combined:<10.4f}")
        print(f"{'Combined Threshold - Aleatoric (50th):':<30} {aleatoric_threshold_combined:<10.4f}")
        print(f"{'Detected anomalies (combined):':<30} {len(anomalies_combined)} / {len(val_dataloader.dataset)}\n")
        
    if save_model_path:
        save_model(model, save_model_path)
    
    return (train_losses, val_losses, train_aleatoric_uncertainties, train_epistemic_uncertainties,
            val_aleatoric_uncertainties, val_epistemic_uncertainties, train_aleatoric_uncertainties_avg, train_epistemic_uncertainties_avg,
            val_aleatoric_uncertainties_avg, val_epistemic_uncertainties_avg, latent_representations, recon_error)


class EAEClient(NumPyClient):
    def __init__(self, cid, model, train_dataloader, val_dataloader, criterion, optimizer, device, config, save_model_path=None):
        self.cid = cid
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.save_model_path = save_model_path

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        lambda_reg = self.config.get("lambda_reg", 0.01)
        num_epochs = self.config.get("num_epochs", 1)
        offset = self.config.get("offset", 2.0)

        total_loss = 0.0
        total_aleatoric_uncertainty_sample = 0.0
        total_epistemic_uncertainty_sample = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            avg_train_loss, train_uncertainties, avg_aleatoric_train, avg_epistemic_train = train(
                self.model, self.criterion, self.optimizer,
                self.train_dataloader, lambda_reg, offset, self.device
            )
            total_loss += avg_train_loss
            total_aleatoric_uncertainty_sample += avg_aleatoric_train
            total_epistemic_uncertainty_sample += avg_epistemic_train

        avg_loss = total_loss / num_epochs
        avg_aleatoric_uncertainty_sample = total_aleatoric_uncertainty_sample / num_epochs
        avg_epistemic_uncertainty_sample = total_epistemic_uncertainty_sample / num_epochs

        print(f"Client {self.cid} - Training Summary:")
        print(f"  Training Loss:               {avg_loss:<10.4f}")
        print(f"  Aleatoric Uncertainty (Avg): {avg_aleatoric_uncertainty_sample:<10.4f}")
        print(f"  Epistemic Uncertainty (Avg): {avg_epistemic_uncertainty_sample:<10.4f}")
        
        if self.save_model_path:
            save_model(self.model, self.save_model_path)

        return self.get_parameters(config), len(self.train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        lambda_reg = self.config.get("lambda_reg", 0.01)
        offset = self.config.get("offset", 2.0)

        avg_val_loss, val_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, recon_error = evaluate_local(
            self.model, self.criterion, self.val_dataloader,
            lambda_reg, offset, self.device
        )

        total_samples = len(self.val_dataloader.dataset)
        
        # Anomaly detection with epistemic uncertainty
        epistemic_threshold, _ = calculate_percentile_thresholds(val_uncertainties, epistemic_percentile=95)
        anomalies_epistemic = [u for u in val_uncertainties if u['epistemic_uncertainty'] > epistemic_threshold]

        epistemic_threshold_combined, aleatoric_threshold_combined = calculate_percentile_thresholds(
            val_uncertainties, epistemic_percentile=95, aleatoric_percentile=50
        )
        anomalies_combined = [
            u for u in val_uncertainties 
            if u['epistemic_uncertainty'] > epistemic_threshold_combined and u['aleatoric_uncertainty'] < aleatoric_threshold_combined
        ]
        
        print("Evaluation Summary:")
        print(f"  Epistemic Percentile Threshold (95th):    {epistemic_threshold:<10.4f}")
        print(f"  Detected {len(anomalies_epistemic):<4} anomalies in validation set using epistemic uncertainty alone out of {total_samples} samples.")
        
        print(f"  Combined Percentile Thresholds:")
        print(f"    Epistemic (95th):                       {epistemic_threshold_combined:<10.4f}")
        print(f"    Aleatoric (50th):                       {aleatoric_threshold_combined:<10.4f}")
        print(f"  Detected {len(anomalies_combined):<4} anomalies in validation set using combined uncertainties out of {total_samples} samples.")

        print(f"  Evaluation Loss:               {avg_val_loss:<10.4f}")
        print(f"  Aleatoric Uncertainty (Avg):   {avg_aleatoric_uncertainty_sample:<10.4f}")
        print(f"  Epistemic Uncertainty (Avg):   {avg_epistemic_uncertainty_sample:<10.4f}")

        return float(avg_val_loss), total_samples, {"loss": float(avg_val_loss)}




def evaluate_saved_model(model_class, model_path, criterion, val_dataloader, lambda_reg, offset, device='cuda', return_latent=False):
    
    # Initializing Model Instances
    model = model_class
    # Load model parameters
    load_model(model, model_path, device)
    
    val_aleatoric_uncertainties = []
    val_epistemic_uncertainties = []
    
    # evaluate
    if return_latent:
        avg_val_loss, val_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, latent_representations, recon_error = evaluate_local(
            model, criterion, val_dataloader, lambda_reg, offset, device, return_latent=True
        )
    else:
        avg_val_loss, val_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, recon_error = evaluate_local(
            model, criterion, val_dataloader, lambda_reg, offset, device, return_latent=False
        )
        latent_representations = None

    val_aleatoric_uncertainties.extend([u['aleatoric_uncertainty'] for u in val_uncertainties])
    val_epistemic_uncertainties.extend([u['epistemic_uncertainty'] for u in val_uncertainties])
    
    # Output evaluation results
    print("\nEvaluation of Saved Model:")
    print(f"{'Validation Loss:':<30} {avg_val_loss:<10.4f}")
    print(f"{'Aleatoric Uncertainty (Avg):':<30} {avg_aleatoric_uncertainty_sample:<10.4f}")
    print(f"{'Epistemic Uncertainty (Avg):':<30} {avg_epistemic_uncertainty_sample:<10.4f}")

    # Uncertainty quartile threshold calculation
    epistemic_threshold, aleatoric_threshold = calculate_percentile_thresholds(val_uncertainties)
    print(f"{'Epistemic Percentile Threshold (95th):':<30} {epistemic_threshold:<10.4f}")
    print(f"{'Aleatoric Percentile Threshold (50th):':<30} {aleatoric_threshold:<10.4f}")

    # Uncertainty-based anomaly detection
    anomalies_epistemic = [u for u in val_uncertainties if u['epistemic_uncertainty'] > epistemic_threshold]
    anomalies_combined = [
        u for u in val_uncertainties 
        if u['epistemic_uncertainty'] > epistemic_threshold and u['aleatoric_uncertainty'] < aleatoric_threshold
    ]

    print(f"{'Detected anomalies (epistemic):':<30} {len(anomalies_epistemic)} / {len(val_dataloader.dataset)}")
    print(f"{'Detected anomalies (combined):':<30} {len(anomalies_combined)} / {len(val_dataloader.dataset)}")

    return avg_val_loss, val_aleatoric_uncertainties, val_epistemic_uncertainties, avg_aleatoric_uncertainty_sample, avg_epistemic_uncertainty_sample, latent_representations, recon_error

