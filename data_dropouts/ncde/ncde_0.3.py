import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import spectrogram
import time # For timing training epochs

# PyTorch CDE specific imports
import torchcde
from torchcde import natural_cubic_spline_coeffs, CubicSpline, cdeint 

# --- Configuration ---
# IMPORTANT: This should be the ABSOLUTE PATH to your 'seizure-prediction' directory.
# Example: '/home/your_user/seizure-prediction'
BASE_DATA_PATH = './seizure-prediction' 

# Define image dimensions
IMG_ROWS, IMG_COLS = 256, 22
NUM_CLASSES = 2 

# Training parameters
BATCH_SIZE = 32 
EPOCHS = 20 
VALIDATION_SPLIT_RATIO = 0.1 
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0 

# Checkpointing configuration
CHECKPOINT_DIR = 'checkpoints' 
# The filename will be dynamically set based on the dropout rate for this specific run
os.makedirs(CHECKPOINT_DIR, exist_ok=True) 

# Determine device for training (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Set Random Seeds for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Can make things slower, but ensures determinism
SEED = 42 
set_seed(SEED)
print(f"Random seed set to {SEED}")
# --- End Set Random Seeds ---

# --- Data Loading and Spectrogram Generation Function (Loads all to memory) ---
# MODIFIED: Added raw_data_zero_dropout_rate parameter and logic
def load_and_process_all_data(patient_num, types, num_segments, root_data_path, fs=5000, segment_length_samples=5000,
                              raw_data_zero_dropout_rate=0.0): # <-- NEW PARAMETER
    """
    Loads raw patient data from .mat files, extracts segments,
    applies raw data zero-dropout (setting selected samples to zero),
    generates spectrograms, and applies normalization.
    """
    all_spectrograms = []
    all_labels = []
    
    patient_mat_files_path = os.path.join(root_data_path, f'Patient_{patient_num}', f'Patient_{patient_num}')

    if not os.path.exists(patient_mat_files_path):
        print(f"Warning: Patient data path not found: {patient_mat_files_path}")
        print("Please ensure your data is correctly placed in the 'seizure-prediction' directory.")
        return [], []

    for i, typ in enumerate(types):
        for j in range(num_segments):
            fl = os.path.join(patient_mat_files_path, '{}_{}.mat'.format(typ, str(j + 1).zfill(4)))
            
            if not os.path.exists(fl):
                print(f"Warning: File not found: {fl}. Skipping this segment.")
                continue

            try:
                data = scipy.io.loadmat(fl)
                typ_prefix_for_key = typ.replace(f'Patient_{patient_num}_', '')
                internal_mat_key = f"{typ_prefix_for_key}_{j + 1}" 
                
                d_array_raw = data[internal_mat_key][0][0][0]
                
                # --- CRITICAL FIX HERE: Ensure only the first channel is used ---
                if d_array_raw.ndim > 1:
                    d_array = d_array_raw[0] 
                    # print(f"Info: d_array in {fl} is multi-dimensional ({d_array_raw.shape}). Taking the first channel for spectrogram.")
                else:
                    d_array = d_array_raw 
                # --- END CRITICAL FIX ---

                total_samples = len(d_array)
                
                for m in range(0, total_samples - segment_length_samples + 1, segment_length_samples):
                    p_secs = d_array[m : m + segment_length_samples].astype(float) # Ensure float for zeroing
                    
                    # --- Apply raw data zero-dropout ---
                    if raw_data_zero_dropout_rate > 0:
                        num_samples_to_zero = int(segment_length_samples * raw_data_zero_dropout_rate)
                        # Randomly select unique indices to set to zero
                        zero_indices = np.random.choice(
                            segment_length_samples, 
                            num_samples_to_zero, 
                            replace=False 
                        )
                        p_secs[zero_indices] = 0.0 # Set selected samples to zero
                    # --- End raw data zero-dropout ---

                    p_f, p_t, p_Sxx = spectrogram(p_secs, fs=fs, return_onesided=False)
                    
                    p_SS = np.log1p(p_Sxx)
                    # Robust normalization, preventing division by zero if all values are zero
                    arr = p_SS[:] / (np.max(p_SS) + 1e-8)
                    
                    all_spectrograms.append(arr)
                    all_labels.append(i)
            except Exception as e:
                print(f"Error processing {fl}: attempted key '{internal_mat_key}' - {e}")
                continue
    return all_spectrograms, all_labels

# --- PyTorch Dataset Class (No temporal dropout in __getitem__ or collate_fn) ---
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, img_rows, img_cols):
        """
        Args:
            spectrograms (list of np.ndarray): Pre-processed spectrograms.
            labels (list of int): Corresponding integer labels.
            img_rows (int): Expected number of rows (frequency bins) in spectrogram.
            img_cols (int): Expected number of columns (time steps) in spectrogram.
        """
        self.img_rows = img_rows
        self.img_cols = img_cols
        
        # Convert to NumPy arrays and then to PyTorch tensors
        # Reshape to (N, C, H, W) -> (N, 1, img_rows, img_cols) for PyTorch Conv2D
        self.X = np.array(spectrograms).reshape(-1, 1, img_rows, img_cols).astype('float32')
        self.Y = np.array(labels) 

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y).long() # Labels should be long for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- Channel Attention Block (unchanged) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        attention = avg_pooled.squeeze(-1).squeeze(-1) 
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention) 

        attention = attention.unsqueeze(-1).unsqueeze(-1)
        attended_features = x * attention 
        return attended_features

# --- Neural CDE ODE Function (unchanged) ---
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z

# --- Main CNN-Neural CDE Model ---
class CNN_NCDE_Model(nn.Module):
    def __init__(self, num_classes=2, cnn_dropout_rate=0.25, ncde_hidden_dim=64, ncde_dropout_rate=0.5):
        super(CNN_NCDE_Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding='same') 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same') 
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn_dropout = nn.Dropout2d(p=cnn_dropout_rate) 

        self.attention_block = ChannelAttention(in_channels=32)

        cnn_out_channels = 32
        freq_bins_after_pooling = IMG_ROWS // 2 # 256 // 2 = 128
        self.ncde_input_dim = cnn_out_channels * freq_bins_after_pooling # 32 * 128 = 4096
        
        self.ncde_hidden_dim = ncde_hidden_dim 

        self.func = CDEFunc(input_channels=self.ncde_input_dim, hidden_channels=self.ncde_hidden_dim)
        self.initial = nn.Linear(self.ncde_input_dim, self.ncde_hidden_dim)
        
        self.ncde_output_dropout = nn.Dropout(p=ncde_dropout_rate) 
        self.output_layer = nn.Linear(self.ncde_hidden_dim, num_classes) 

    def forward(self, x):
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.cnn_dropout(x)
        
        x = self.attention_block(x)
        
        # Reshape for CDE: (batch, sequence_length, input_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # Permute to (batch, time, channels, freq)
        x_cde_input = x.view(
            x.size(0), 
            x.size(1), 
            -1
        ) # Reshape to (batch, time, flattened_features)
        
        # Define time points for the CDE based on the actual number of time bins
        t_cde = torch.linspace(0, x_cde_input.size(1) - 1, x_cde_input.size(1), device=x_cde_input.device)
        
        # Compute natural cubic spline coefficients
        coeffs = natural_cubic_spline_coeffs(x_cde_input, t=t_cde)
        spline = CubicSpline(coeffs, t=t_cde)
        
        # Initial state for CDE, derived from the first point of the spline
        evaluated_z0_input = spline.evaluate(t_cde[0])
        z0 = self.initial(evaluated_z0_input) 

        # Time span for CDE integration (from start to end of the sequence)
        t_span = torch.stack([t_cde[0], t_cde[-1]]) 
        
        # Adjoint parameters for backpropagation
        adjoint_params = tuple(self.func.parameters()) + (coeffs,)

        # Integrate CDE
        z_T = cdeint(
            z0=z0,
            func=self.func, 
            X=spline, 
            t=t_span, 
            atol=1e-2, 
            rtol=1e-2,
            adjoint_params=adjoint_params 
        )
        
        # Output from the last time step of the CDE integration
        x = self.ncde_output_dropout(z_T[:, -1, :]) 
        
        outputs = self.output_layer(x)
        
        return outputs

# --- Training and Evaluation Functions (unchanged, includes checkpoint loading in train_model) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                checkpoint_dir, checkpoint_filename):
    """
    Trains the PyTorch model with checkpointing.
    Saves the model with the best validation accuracy.
    """
    model.train() 
    best_val_accuracy = -1.0 

    full_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    start_epoch = 0
    if os.path.exists(full_checkpoint_path):
        checkpoint = torch.load(full_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_val_accuracy = checkpoint.get('val_accuracy', -1.0) 
        print(f"Checkpoint loaded. Resuming from Epoch {start_epoch} with Val Acc: {best_val_accuracy:.4f}")

    for epoch in range(start_epoch, epochs): 
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() 
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward() 
            optimizer.step() 

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1) 
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        end_time = time.time()
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        model.train() 

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} - "
              f"Time: {end_time - start_time:.2f}s")

        # Save checkpoint if current validation accuracy is better than the best seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'val_accuracy': val_accuracy,
            }, full_checkpoint_path)
            print(f"   Checkpoint saved: {full_checkpoint_path} (Val Acc: {val_accuracy:.4f})")

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the PyTorch model.
    """
    model.eval() 
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): 
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting PyTorch CNN-Neural CDE Model Training with Raw Data Zero-Dropout Experiment...")
    
    # --- CONFIGURE THE DESIRED DROPOUT RATE HERE FOR THIS RUN ---
    # Set this value to 0.0, 0.3, 0.5, or 0.7 for your individual runs.
    RAW_DATA_ZERO_DROPOUT_RATE = 0.3 # Example: Setting to 0.3 for a dropout run
    # -----------------------------------------------------------

    set_seed(SEED) # Ensure reproducibility for this specific run
    print(f"\n--- Running experiment with RAW data zero-dropout rate: {RAW_DATA_ZERO_DROPOUT_RATE:.2f} ---")

    # 1. Load ALL Data into memory first, applying raw data zero-dropout
    print("Loading and processing Patient 1 data into memory...")
    types_p1 = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']
    all_X1, all_Y1 = load_and_process_all_data(1, types_p1, 18, BASE_DATA_PATH, 
                                                raw_data_zero_dropout_rate=RAW_DATA_ZERO_DROPOUT_RATE) 
    print(f"Patient 1 data processed: {len(all_X1)} samples.")

    print("Loading and processing Patient 2 data into memory...")
    types_p2 = ['Patient_2_interictal_segment', 'Patient_2_preictal_segment']
    all_X2, all_Y2 = load_and_process_all_data(2, types_p2, 18, BASE_DATA_PATH, 
                                                raw_data_zero_dropout_rate=RAW_DATA_ZERO_DROPOUT_RATE) 
    print(f"Patient 2 data processed: {len(all_X2)} samples.")

    # Combine data from both patients
    all_X_combined = all_X1 + all_X2
    all_Y_combined = all_Y1 + all_Y2
    print(f"Combined processed data: {len(all_X_combined)} samples.")

    # Shuffle the combined data (crucial for random split)
    print("Shuffling combined data...")
    combined_data_list = list(zip(all_X_combined, all_Y_combined))
    random.shuffle(combined_data_list)
    all_X_shuffled, all_Y_shuffled = zip(*combined_data_list)
    all_X_shuffled = list(all_X_shuffled)
    all_Y_shuffled = list(all_Y_shuffled)
    print("Data shuffled.")

    # Create a single PyTorch Dataset from the shuffled, in-memory data
    full_dataset = SpectrogramDataset(all_X_shuffled, all_Y_shuffled, IMG_ROWS, IMG_COLS)
    print(f"Total dataset size: {len(full_dataset)} samples.")

    # Split into train, validation, and test sets
    original_train_size = 42000
    if original_train_size > len(full_dataset):
        print(f"Warning: Original train size ({original_train_size}) exceeds total dataset size ({len(full_dataset)}). Adjusting split.")
        original_train_size = len(full_dataset) 

    train_val_dataset = torch.utils.data.Subset(full_dataset, range(original_train_size))
    test_dataset = torch.utils.data.Subset(full_dataset, range(original_train_size, len(full_dataset)))

    train_size = int(len(train_val_dataset) * (1 - VALIDATION_SPLIT_RATIO))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("\nDataLoaders created:")
    print(f"Train DataLoader batches: {len(train_loader)}")
    print(f"Validation DataLoader batches: {len(val_loader)}")
    print(f"Test DataLoader batches: {len(test_loader)}")

    # 2. Initialize Model, Loss, and Optimizer for THIS specific run
    model = CNN_NCDE_Model(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters())

    print("\nModel Summary:")
    print(model) 

    # 3. Train the Model for the current dropout rate
    # Dynamic checkpoint filename to distinguish runs
    checkpoint_filename_for_this_run = f'best_cnn_ncde_raw_data_dropout_{int(RAW_DATA_ZERO_DROPOUT_RATE*100)}.pth'
    print(f"\nStarting model training for {EPOCHS} epochs with RAW data zero-dropout rate {RAW_DATA_ZERO_DROPOUT_RATE:.2f}...")
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                        epochs=EPOCHS, device=DEVICE, 
                        checkpoint_dir=CHECKPOINT_DIR, checkpoint_filename=checkpoint_filename_for_this_run)
    print("Model training complete.")

    # 4. Evaluate the Model on Test Data for the current dropout rate
    best_model_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename_for_this_run)
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model for evaluation from {best_model_path}")
    print(f"\nEvaluating model on test data for RAW data zero-dropout rate {RAW_DATA_ZERO_DROPOUT_RATE:.2f}...")
    model.eval()
    model.to(DEVICE) 
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    print(f"\n--- Results for Raw Data Zero-Dropout Rate: {RAW_DATA_ZERO_DROPOUT_RATE:.2f} ---")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print("\nPyTorch CNN-Neural CDE (Raw Data Zero-Dropout Experiment) process finished.")
