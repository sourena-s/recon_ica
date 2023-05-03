import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import pandas as pd
import torch.optim as optim


device = torch.device("cuda:0")


class CustomAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CustomAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
# Lock the decoder weights
        self.decoder.weight.requires_grad = False
       # Initialize encoder weights randomly
        nn.init.normal_(self.encoder.weight, mean=0, std=0.01)
        # Initialize decoder weights as transpose of the encoder weights
        self.decoder.weight.data = self.encoder.weight.data.t()
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def update_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

def kurtosis_cost(encoded):
    kurt = torch.mean((encoded**4), dim=0) # - 3
    return torch.sum(torch.abs(kurt))

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05):
        super(CustomBatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps=eps)
    def forward(self, x):
        return self.batch_norm(x)

# Load 4D NIfTI file
nifti_file = '/data/clusterfs/lag/users/sousoh/ukbb/genetic/design-matrix-dMRI/merged-fixels-QC-passed-all-varnorm.nii.gz'
mask_file= '/data/clusterfs/lag/users/sousoh/ukbb/pilot-1k/t1-syn-template/qc-masks/stage_11_mask_roi_iso2.nii.gz'
do_masking=False

scan = nib.load(nifti_file)
#print(scan.header)
data = scan.get_fdata()

mask_nifti = nib.load(mask_file)
mask_data = mask_nifti.get_fdata()

if len(data.shape) == 4:
# Flatten the spatial dimensions
    print(f"Data is 4 dimensional, assuming samples run along the fourth axis.")
    flattened_data = data.reshape((-1, data.shape[3]))
    flattened_mask = mask_data.reshape(-1)
    is_4d = True
elif len(data.shape) == 2:
    print(f"Data is 2 dimensional, assuming samples run along the second axis.")
    flattened_data = data
    flattened_mask = mask_data.reshape(-1)
    is_4d = False
else:
    raise ValueError("Input NIfTI data must be either 2D or 4D.")


if do_masking:
# Remove elements with a value of zero in the mask
    masked_data = flattened_data[flattened_mask > 0]
else:
    masked_data = flattened_data

# Remove elements with zero variability along the fourth dimension
nonzero_var_indices = np.var(masked_data, axis=1) > 0
final_input_data = masked_data[nonzero_var_indices]



# User-defined flags
demean = True
variance_normalise = True
batch_normalise = False
non_gaussianity_hyperparam = 0.1

# Preprocess the data
if demean:
    final_input_data -= np.mean(final_input_data, axis=1, keepdims=True)

if variance_normalise:
    final_input_data /= np.std(final_input_data, axis=1, keepdims=True)

# Convert flattened data to PyTorch tensor
input_data = torch.tensor(final_input_data, dtype=torch.float32)

# Training loop
num_epochs = 100
batch_size = 64  # Set the desired batch size

input_dim = flattened_data.shape[1]  # Set input dimension based on the flattened data
latent_dim = 2000

autoencoder = CustomAutoencoder(input_dim, latent_dim)
autoencoder = autoencoder.to(device)

learning_rate = 0.001
optimizer = optim.Adam(autoencoder.encoder.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Add batch normalization if the flag is set
if batch_normalise:
    batch_norm = CustomBatchNorm(input_dim)

loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size, pin_memory=True)

for epoch in range(num_epochs):
    #for i in range(0, input_data.shape[0], batch_size):
        #batch_input_data = input_data[i:i + batch_size]
        #batch_input_data = batch_input_data.to(device)
    for batch in loader:
        batch_input_data = batch.to(device)

        # Apply batch normalization if the flag is set
        if batch_normalise:
            batch_input_data = batch_norm(batch_input_data)
        optimizer.zero_grad()
        output = autoencoder(batch_input_data)
        #reconstruction_loss = nn.L1Loss()(output, batch_input_data)
        reconstruction_loss = criterion(output, batch_input_data)
#       independence_loss = non_gaussianity_hyperparam * kurtosis_cost(autoencoder.encoder(batch_input_data))
#        loss = reconstruction_loss + non_gaussianity_hyperparam * independence_loss 
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()
        autoencoder.update_decoder_weights()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Obtain the decoder weights
decoder_weights = autoencoder.decoder.weight.data.cpu().numpy()



rearranged_weights = np.zeros((flattened_mask.shape[0], decoder_weights.shape[1]))

mask_indices = np.nonzero(flattened_mask > 0)[0]
rearranged_weights[mask_indices[nonzero_var_indices]] = decoder_weights

# Reshape rearranged weights to the original input data shape
if input_nifti.ndim == 4:
    final_weights_data = rearranged_weights.reshape((*input_data.shape[:-1], decoder_weights.shape[1]))
else:
    final_weights_data = rearranged_weights.reshape((*input_data.shape, decoder_weights.shape[1]))























if is_4d:
# Rearrange the weights to 3D data
    decoder_weights_3d = decoder_weights.reshape((-1, *data.shape[:3]))

# Concatenate 3D data along a new dimension (axis=0)
    concatenated_weights = np.stack(decoder_weights_3d, axis=-1)

# Save the concatenated 3D data as a 4D NIfTI file
    output_nifti = nib.Nifti1Image(concatenated_weights, img.affine)
    nib.save(output_nifti, 'path/to/output_nifti_file.nii')
else:
    # Save the concatenated weights as a 2D NIfTI file
    output_nifti = nib.Nifti1Image(decoder_weights, img.affine)
    nib.save(output_nifti, 'path/to/output_nifti_file_2D.nii')


# Get the activities of the latent variables per each sample
encoded_data = autoencoder.encoder(input_data).data.cpu().numpy()

# Save the activities as a table (dataframe)
activity_table = pd.DataFrame(encoded_data, columns=[f'LV_{i+1}' for i in range(latent_dim)])
activity_table.to_csv('path/to/output_activity_table.csv', index=False)

