# Required Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import requests
import zipfile
import io
from tqdm import tqdm
import h5py

MAX_STEPS = 10

def download_radarscenes_dataset(url, extract_to='.'):
    print("Downloading RadarScenes dataset...")
    # Check if the directory already exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)  # Create the directory if it doesn't exist
    else:
        print(f"Directory '{extract_to}' already exists. Skipping download and extraction.")
        return

    # Download the file from the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    # Total size in bytes
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    temp_buffer = io.BytesIO()

    print("Downloading...")
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as t:
        for data in response.iter_content(block_size):
            t.update(len(data))
            temp_buffer.write(data)

    # Extract the content of the ZIP file with a progress bar
    print("Extracting...")
    temp_buffer.seek(0)  # Go back to the beginning of the buffer
    with zipfile.ZipFile(temp_buffer) as z:
        file_list = z.namelist()
        for file_name in tqdm(file_list, desc="Extracting"):
            z.extract(file_name, path=extract_to)

    print(f"Files extracted to '{extract_to}'")

class RadarScenesDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.files = []
        self.data = []

        # Loop over all sequence_x folders
        for sequence_folder in sorted(os.listdir(base_path)):
            sequence_path = os.path.join(base_path, sequence_folder)
            radar_data_file = os.path.join(sequence_path, 'radar_data.h5')

            # Add file path to list if it exists
            if os.path.exists(radar_data_file):
                self.files.append(radar_data_file)
            else:
                print(f"File not found: {radar_data_file}")

        # Load and preprocess data from all files
        for radar_data_file in self.files:
            with h5py.File(radar_data_file, 'r') as f:
                point_cloud_data = {
                    'rcs': f['radar_data']['rcs'][:],
                    'range_sc': f['radar_data']['range_sc'][:],
                    'azimuth_sc': f['radar_data']['azimuth_sc'][:],
                    'vr': f['radar_data']['vr'][:],
                    'vr_compensated': f['radar_data']['vr_compensated'][:],
                    'x_cc': f['radar_data']['x_cc'][:],
                    'y_cc': f['radar_data']['y_cc'][:],
                    'x_seq': f['radar_data']['x_seq'][:],
                    'y_seq': f['radar_data']['y_seq'][:],
                    'uuid': f['radar_data']['uuid'][:],
                    'track_id': f['radar_data']['track_id'][:],
                    'label_id': f['radar_data']['label_id'][:]
                }
                self.data.append(point_cloud_data)

        print("Dataset Loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]

        # Combine the numpy arrays into a single numpy array
        features = np.stack([
            point_cloud['x_cc'],
            point_cloud['y_cc'],
            point_cloud['vr'],
            point_cloud['vr_compensated'],
            point_cloud['range_sc'],
            point_cloud['azimuth_sc'],
            point_cloud['x_seq'],
            point_cloud['y_seq'],
            point_cloud['rcs']
        ], axis=-1).astype(np.float32)

        # Apply downsampling
        num_points = 10000
        indices = np.random.choice(features.shape[0], num_points, replace=False)
        features = features[indices]
        label = point_cloud['label_id'][indices]

        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return features, label


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def filter_labels(features, labels, label_to_drop):
    filtered_features = []
    filtered_labels = []

    # Flatten features and labels if they are batched
    for i in range(len(labels)):
        if isinstance(labels[i], torch.Tensor):
            batch_labels = labels[i]
        else:
            batch_labels = torch.tensor([labels[i]])

        if isinstance(features[i], torch.Tensor):
            batch_features = features[i]
        else:
            batch_features = torch.tensor([features[i]])

        # Filter out unwanted labels
        mask = batch_labels != label_to_drop
        filtered_features.extend(batch_features[mask].tolist())
        filtered_labels.extend(batch_labels[mask].tolist())

    # Convert lists back to tensors
    features_tensor = torch.tensor(filtered_features, dtype=torch.float32)  # Adjust dtype if necessary
    labels_tensor = torch.tensor(filtered_labels, dtype=torch.long)

    return CustomDataset(features_tensor, labels_tensor)

def process_dataset(radar_dataset, label_to_drop=11):
    features = []
    labels = []

    for feature, label in radar_dataset:
        if isinstance(label, torch.Tensor):
            labels.append(label)
            features.append(feature)
        else:
            labels.append(torch.tensor([label]))  # Wrap non-tensor labels in a tensor
            features.append(feature)

    # Flatten lists of tensors
    features_tensor = torch.cat(features)
    labels_tensor = torch.cat(labels)

    # Apply filtering to drop specific labels
    filtered_dataset = filter_labels(features_tensor, labels_tensor, label_to_drop)
    
    return filtered_dataset

# RNN Model for RadarScenes dataset (Radar)
class RadarRNNModel(nn.Module):
    def __init__(self):
        super(RadarRNNModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=9, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 12)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize h0 and c0 for LSTM1
        h0_1 = torch.zeros(2, batch_size, 128)  # 2 layers, batch size, hidden size
        c0_1 = torch.zeros(2, batch_size, 128)
        
        # Initialize h0 and c0 for LSTM2
        h0_2 = torch.zeros(2, batch_size, 256)  # 2 layers, batch size, hidden size
        c0_2 = torch.zeros(2, batch_size, 256)
        
        # Flatten the input if needed and reshape back for LSTM
        x = x.view(batch_size, -1, x.size(-1))  # Ensure shape is [batch_size, seq_length, input_size]

        # LSTM layers with initial hidden and cell states
        x, _ = self.lstm1(x, (h0_1, c0_1))
        x, _ = self.lstm2(x, (h0_2, c0_2))

        # Use the last time step output for classification
        x = x[:, -1, :]

        # Apply fully connected layers with batch normalization
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_model_radar(model, dataloader, criterion, optimizer, device):
    print("Training ...")   
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    correct = 0
    total = 0

    # Limit the number of steps
    max_steps = MAX_STEPS
    step_count = 0

    for features, labels in dataloader:
        if step_count >= max_steps:
            break  # Stop after two steps
        # Ensure features have the correct shape
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence length dimension

        # Normalize inputs to prevent gradient explosion
        features = (features - features.mean()) / features.std()

        optimizer.zero_grad()
        outputs = model(features)

        # Flatten outputs and labels for loss computation
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)

        # Match outputs and labels batch sizes if they differ
        if outputs.size(0) != labels.size(0):
            min_size = min(outputs.size(0), labels.size(0))
            outputs = outputs[:min_size]
            labels = labels[:min_size]

        # Compute loss
        loss = criterion(outputs, labels)
        if torch.isnan(loss).any():  # Check for NaN in loss
            print("NaN detected in loss, skipping update")
            continue  # Skip this batch

        loss.backward()

        # Apply gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=11)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

def validate_model_radar(model, dataloader, criterion, device):
    print('Start validating ...')
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy