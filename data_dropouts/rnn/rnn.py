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

# --- Configuration ---
# IMPORTANT: Update this to the ABSOLUTE PATH of your 'seizure-prediction' directory.
# Example: '/home/your_user/seizure-prediction'
BASE_DATA_PATH = './seizure-prediction' 

# Define image dimensions
IMG_ROWS, IMG_COLS = 256, 22
NUM_CLASSES = 2 

# Training parameters
BATCH_SIZE = 32 # Using the batch size from your CDE notebook for consistency
EPOCHS = 20 # Can be adjusted
VALIDATION_SPLIT_RATIO = 0.1 
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0 # Use half CPU cores, or 0 if not defined/safe

# Checkpointing configuration
CHECKPOINT_DIR = 'checkpoints' 
# Updated filename to reflect this specific model version (RNN instead of LSTM)
CHECKPOINT_FILENAME = 'best_cnn_rnn_no_temp_dropout_model.pth' 
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
def load_and_process_all_data(patient_num, types, num_segments, root_data_path, fs=5000, segment_length_samples=5000):
    """
    Loads raw patient data from .mat files, extracts segments, generates spectrograms,
    and applies normalization. Returns all data in memory.
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
                    d_array = d_array_raw # Already 1D
                # --- END CRITICAL FIX ---

                total_samples = len(d_array)
                
                for m in range(0, total_samples - segment_length_samples + 1, segment_length_samples):
                    p_secs = d_array[m : m + segment_length_samples]
                    
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
        self.img_rows = img_rows
        self.img_cols = img_cols
        
        # Reshape to (N, C, H, W) -> (N, 1, img_rows, img_cols) for PyTorch Conv2D
        # Spectrograms are already (H, W) = (256, 22)
        self.X = np.array(spectrograms).reshape(-1, 1, img_rows, img_cols).astype('float32')
        self.Y = np.array(labels) 

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y).long() # Labels should be long for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- Attention Block (from your original LSTM model) ---
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.global_avg_pool(x)
        attention = attention.view(attention.size(0), -1) 
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.unsqueeze(2).unsqueeze(3) 
        out = x * attention
        return out

# --- CNN-RNN Model with Corrected Temporal Input (No Temporal Dropout) ---
class CNNRNNWithAttention(nn.Module):
    def __init__(self, img_rows, img_cols, num_classes):
        super(CNNRNNWithAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding='same') 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same') 
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25) # Standard dropout on features after pooling

        self.attention_block = AttentionBlock(in_channels=32) # Using your original AttentionBlock

        # Calculate input size for RNN based on corrected reshape
        cnn_out_channels = 32
        freq_bins_after_pooling = img_rows // 2 # 256 // 2 = 128
        
        self.rnn_input_size = cnn_out_channels * freq_bins_after_pooling # 32 * 128 = 4096
        
        # --- CHANGED: Using nn.RNN instead of nn.LSTM ---
        self.rnn = nn.RNN(input_size=self.rnn_input_size, hidden_size=64, batch_first=True) 
        # --- END CHANGE ---

        self.dropout2 = nn.Dropout(0.5) # Standard dropout on RNN output
        
        self.fc_out = nn.Linear(64, num_classes) 

    def forward(self, x):
        # CNN Feature Extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x) # Shape: (batch, 32, 128, 11)
        x = self.dropout1(x) 
        
        x = self.attention_block(x) # Shape: (batch, 32, 128, 11)
        
        # --- Correct Reshape for sequential RNN processing ---
        # Current x shape: (batch_size, channels, freq_bins, time_bins) e.g., (B, 32, 128, 11)
        # Desired x_rnn_input shape: (batch_size, time_bins, channels * freq_bins) e.g., (B, 11, 4096)
        x = x.permute(0, 3, 1, 2).contiguous() # Permute to (batch, time_steps, channels, freq_bins)
        x_rnn_input = x.view(
            x.size(0), 
            x.size(1), # This is the time dimension (11)
            -1 # This flattens channels * freq_bins (32 * 128 = 4096)
        ) 
        # --- End Reshape ---
        
        # --- CHANGED: Calling self.rnn instead of self.lstm ---
        rnn_out, _ = self.rnn(x_rnn_input) 
        # --- END CHANGE ---
        
        # Take the output from the LAST time step of the RNN for classification
        x = rnn_out[:, -1, :] # Shape: (batch, hidden_size) e.g., (B, 64)
        
        x = self.dropout2(x)
        
        out = self.fc_out(x) 
        
        return out

# --- Training and Evaluation Functions (from your CDE model, general purpose) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                checkpoint_dir, checkpoint_filename):
    """
    Trains the PyTorch model with checkpointing.
    Saves the model with the best validation accuracy.
    """
    model.train() # Set model to training mode
    best_val_accuracy = -1.0 # Initialize with a low value

    full_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    # If resuming, load the best_val_accuracy from the loaded checkpoint
    if os.path.exists(full_checkpoint_path):
        checkpoint = torch.load(full_checkpoint_path, map_location=device)
        best_val_accuracy = checkpoint.get('val_accuracy', -1.0) # Get existing best val acc

    for epoch in range(epochs):
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
        model.train() # Set back to train mode after evaluation

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
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
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

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads a model and optimizer state from a checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0, -1.0 # Return start_epoch = 0, best_val_accuracy = -1.0 (so first save is always made)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1 # Start from the next epoch
    best_val_accuracy = checkpoint['val_accuracy']
    print(f"Checkpoint loaded. Resuming from Epoch {epoch} with Val Acc: {best_val_accuracy:.4f}")
    return epoch, best_val_accuracy

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting PyTorch CNN-RNN Model Training (Corrected Temporal Input, No Temporal Dropout)...")
    
    # 1. Load ALL Data into memory first (similar to original TF notebook)
    print("Loading and processing Patient 1 data into memory...")
    types_p1 = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']
    all_X1, all_Y1 = load_and_process_all_data(1, types_p1, 18, BASE_DATA_PATH)
    print(f"Patient 1 data processed: {len(all_X1)} samples.")

    print("Loading and processing Patient 2 data into memory...")
    types_p2 = ['Patient_2_interictal_segment', 'Patient_2_preictal_segment']
    all_X2, all_Y2 = load_and_process_all_data(2, types_p2, 18, BASE_DATA_PATH)
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
    # Convert back to lists for SpectrogramDataset
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

    # Create the train_val and test sets from the fully shuffled data
    train_val_dataset = torch.utils.data.Subset(full_dataset, range(original_train_size))
    test_dataset = torch.utils.data.Subset(full_dataset, range(original_train_size, len(full_dataset)))

    # Further split the train_val_dataset into training and validation
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

    # 2. Initialize Model, Loss, and Optimizer
    # --- CHANGED: Instantiating CNNRNNWithAttention model ---
    model = CNNRNNWithAttention(IMG_ROWS, IMG_COLS, NUM_CLASSES).to(DEVICE)
    # --- END CHANGE ---
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters())

    print("\nModel Summary:")
    print(model) 

    # 3. Load Checkpoint (if exists)
    checkpoint_path_full = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    start_epoch, best_val_accuracy = load_checkpoint(model, optimizer, checkpoint_path_full, DEVICE)
    
    # Adjust EPOCHS if resuming
    remaining_epochs = EPOCHS - start_epoch
    if remaining_epochs <= 0:
        print("Model already trained for the specified number of epochs or more. Skipping training.")
    else:
        # 4. Train the Model
        print(f"\nStarting model training for {remaining_epochs} epochs...")
        train_model(model, train_loader, val_loader, criterion, optimizer, 
                            epochs=remaining_epochs, device=DEVICE, 
                            checkpoint_dir=CHECKPOINT_DIR, checkpoint_filename=CHECKPOINT_FILENAME)
        print("Model training complete.")

    # 5. Evaluate the Model on Test Data
    print("\nEvaluating model on test data...")
    model.eval()
    model.to(DEVICE) # Ensure model is on the correct device for evaluation
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    print("\nPyTorch CNN-RNN (No Temporal Dropout) conversion process finished.")