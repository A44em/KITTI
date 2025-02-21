# Required Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

MAX_STEPS = 10

# Dataset download function (placeholder for actual download)
def download_kitti_dataset_img(url, extract_to='.'):
    print("Downloading camera KITTI dataset...")

    # Check if the directory already exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)  # Create the directory if it doesn't exist
    else:
        print(f"Directory '{extract_to}' already exists. Skipping download and extraction.")
        return

    # Local file name to save the downloaded file
    local_zip_path = 'data_object_image_2.zip'

    # Download the file with progress bar
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
        block_size = 1024  # 1 Kilobyte

        # Create a progress bar using tqdm
        with open(local_zip_path, 'wb') as file, tqdm(
            desc=local_zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        print(f"Downloaded {local_zip_path} successfully.")
    else:
        print(f"Failed to download file, status code: {response.status_code}")
        return

    # Extract the downloaded ZIP file
    if os.path.exists(local_zip_path):
        print(f"Extracting {local_zip_path}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            extract_path = '/home/a44em_3li/KITTI_img'
            zip_ref.extractall(extract_path)
            print(f"Extracted data to {extract_path}.")
    else:
        print(f"Zip file {local_zip_path} not found.")

def download_kitti_dataset_lidar(url, extract_to='.'):
    print("Downloading lidar KITTI dataset...")
    
    # Check if the directory already exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)  # Create the directory if it doesn't exist
    else:
        print(f"Directory '{extract_to}' already exists. Skipping download and extraction.")
        return

    # Local file name to save the downloaded file
    local_zip_path = 'data_object_velodyne.zip'

    # Download the file with progress bar
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
        block_size = 1024  # 1 Kilobyte

        # Create a progress bar using tqdm
        with open(local_zip_path, 'wb') as file, tqdm(
            desc=local_zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        print(f"Downloaded {local_zip_path} successfully.")
    else:
        print(f"Failed to download file, status code: {response.status_code}")
        return

    # Extract the downloaded ZIP file
    if os.path.exists(local_zip_path):
        print(f"Extracting {local_zip_path}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            extract_path = '/home/a44em_3li/KITTI_lidar'  # Directory where data will be extracted
            zip_ref.extractall(extract_path)
            print(f"Extracted data to {extract_path}.")
    else:
        print(f"Zip file {local_zip_path} not found.")


def download_kitti_dataset_labels(url, extract_to='.'):
    print("Downloading labels KITTI dataset...")
    
    # Check if the directory already exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)  # Create the directory if it doesn't exist
    else:
        print(f"Directory '{extract_to}' already exists. Skipping download and extraction.")
        return

    # Local file name to save the downloaded file
    local_zip_path = 'data_object_label_2.zip'

    # Download the file with progress bar
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
        block_size = 1024  # 1 Kilobyte

        # Create a progress bar using tqdm
        with open(local_zip_path, 'wb') as file, tqdm(
            desc=local_zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

        print(f"Downloaded {local_zip_path} successfully.")
    else:
        print(f"Failed to download file, status code: {response.status_code}")
        return

    # Extract the downloaded ZIP file
    if os.path.exists(local_zip_path):
        print(f"Extracting {local_zip_path}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            extract_path = '/home/a44em_3li/KITTI_labels'
            zip_ref.extractall(extract_path)
            print(f"Extracted data to {extract_path}.")
    else:
        print(f"Zip file {local_zip_path} not found.")

# Custom Dataset Class (Placeholder for actual loading)
class KITTIDataset(Dataset):
    def __init__(self, image_dir, lidar_dir, label_dir, transform=None, class_mapping=None):
        self.image_dir = image_dir
        self.lidar_dir = lidar_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.lidar_files = sorted(os.listdir(lidar_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transform
        self.class_mapping = class_mapping if class_mapping else {}

    def __len__(self):
        # Use the length of the shortest set (images, lidar, or labels)
        return min(len(self.image_files), len(self.lidar_files), len(self.label_files))

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)
        else:
            # Resize and convert to tensor if no transform is provided
            resize_transform = transforms.Resize((128, 128))  # Example size
            image = resize_transform(image)
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load LiDAR data (assuming it's in binary format)
        lidar_path = os.path.join(self.lidar_dir, self.lidar_files[idx])
        lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # Assuming 4 values per LiDAR point (x, y, z, intensity)
        lidar_data = torch.tensor(lidar_data, dtype=torch.float32)

        # Load labels
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                # Example format: 'Car 0.00 0 1.57 598.32 156.40 626.53 189.25 1.57 1.48 3.69 -1.47 1.67 46.00 -1.61'
                elements = line.strip().split(' ')
                object_type = elements[0]
                if object_type in self.class_mapping:  # Ignore 'DontCare' and unmapped classes
                    class_id = self.class_mapping[object_type]
                    
                    # Parse the bounding box coordinates
                    bbox = list(map(float, elements[4:8]))  # (x_min, y_min, x_max, y_max)
                    
                    # Parse the object's 3D size and position if necessary (e.g., for 3D object detection)
                    dimensions = list(map(float, elements[8:11]))  # height, width, length
                    location = list(map(float, elements[11:14]))   # x, y, z
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': bbox,
                        'dimensions': dimensions,
                        'location': location
                    })

        # Convert labels into tensors for multi-object support
        # If no objects are found, return an empty tensor for labels
        if labels:
            # Assuming 'labels' is a list of dictionaries containing the keys 'class_id', 'bbox', 'dimensions', and 'location'
            class_ids = torch.tensor([obj['class_id'] for obj in labels], dtype=torch.long)  # Directly use obj['class_id'] as it's an integer
            bboxes = torch.tensor([obj['bbox'] for obj in labels], dtype=torch.float32)        # Assuming obj['bbox'] is a list/array of floats
            dimensions = torch.tensor([obj['dimensions'] for obj in labels], dtype=torch.float32)  # Assuming obj['dimensions'] is a list/array of floats
            locations = torch.tensor([obj['location'] for obj in labels], dtype=torch.float32)  # Assuming obj['location'] is a list/array of floats
        else:
            class_ids = torch.empty(0, dtype=torch.long)
            bboxes = torch.empty(0, 4, dtype=torch.float32)
            dimensions = torch.empty(0, 3, dtype=torch.float32)
            locations = torch.empty(0, 3, dtype=torch.float32)

        return image, lidar_data, class_ids, bboxes, dimensions, locations
    

# Define the class mapping
class_mapping = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1  # We may ignore DontCare regions
}

class KITTICNNModel(nn.Module):
    def __init__(self, max_lidar_points=120000):  # Specify maximum LiDAR points to pad
        super(KITTICNNModel, self).__init__()
        self.max_lidar_points = max_lidar_points  # Max points to pad for LiDAR data
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Dummy input to determine the output size after convolutions
        self._initialize_fc1()

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size + 4 * self.max_lidar_points, 256)  # Adjust based on padded LiDAR points
        self.fc2 = nn.Linear(256, 12)  # Assuming 12 classes

    def _initialize_fc1(self):
        # Use a dummy input to automatically calculate the correct output size for fc1
        dummy_input = torch.zeros(1, 3, 128, 128)  # Use 128x128 or your input size
        dummy_out = self.conv2(self.conv1(dummy_input))
        self.conv_output_size = dummy_out.view(1, -1).size(1)  # Flatten and get the feature size

    def forward(self, x, lidar=None):  # Accept lidar input
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        if lidar is not None:
            # Pad each LiDAR tensor to the maximum specified points
            padded_lidar = []
            for lidar_sample in lidar:
                num_points = lidar_sample.size(0)
                if num_points < self.max_lidar_points:
                    # Pad with zeros if the number of points is less than the maximum
                    pad_size = (0, 0, 0, self.max_lidar_points - num_points)
                    lidar_sample = F.pad(lidar_sample, pad_size)
                else:
                    # Truncate if the number of points exceeds the maximum
                    lidar_sample = lidar_sample[:self.max_lidar_points, :]
                
                padded_lidar.append(lidar_sample)
            
            # Stack the padded tensors along the batch dimension
            lidar_features = torch.stack(padded_lidar, dim=0).view(len(lidar), -1)  # Flatten
            
            # Concatenate LiDAR features with image features
            x = torch.cat((x, lidar_features), dim=1)
            
        return self.fc2(self.fc1(x))  # Pass through fc1 then fc2 for final output
    
# Training and validation functions
def train_model(model, dataloader, criterion, optimizer, device):
    print('Start training ...')
    #print(len(dataloader))
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Limit the number of steps
    max_steps = MAX_STEPS
    step_count = 0

    for data in dataloader:
        if step_count >= max_steps:
            break  # Stop after two steps
        # Unpack the data according to the returned components
        images, lidars, class_ids, bboxes, dimensions, locations = data
        
        # Move inputs to the appropriate device
        images = images.to(device)
        lidars = [lidar.to(device) for lidar in lidars]  # Move each LiDAR tensor to the device
        # Pad class_ids to a fixed length
        max_labels=10
        # Convert each class_id to a list and pad it as needed
        padded_class_ids = [cls_id.tolist() + [0] * (max_labels - len(cls_id)) if len(cls_id) < max_labels else cls_id.tolist()[:max_labels] for cls_id in class_ids]
        class_ids = torch.tensor(padded_class_ids).to(device)
        # Convert padded class_ids to 1D by selecting the first label in each sample
        primary_class_ids = torch.tensor([cls_id[0] for cls_id in padded_class_ids]).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, lidars)  # Assuming your model takes both images and LiDAR data as input
        
        # Compute loss (assuming you're only interested in class_ids for the loss)
        loss = criterion(outputs, primary_class_ids)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)  # Get the predicted class
        total += primary_class_ids.size(0)
        correct += predicted.eq(primary_class_ids).sum().item()
        # Increment step count
        step_count += 1
        # print(f"Step {step_count}/{max_steps} completed")

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

def validate_model(model, dataloader, criterion, device):
    print('Start validating ...')
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Limit the number of steps
    max_steps = MAX_STEPS
    step_count = 0

    with torch.no_grad():
        for data in dataloader:
            if step_count >= max_steps:
                break  # Stop after two steps
            # Unpack the data according to the returned components
            images, lidars, class_ids, bboxes, dimensions, locations = data
            
            # Move inputs to the appropriate device
            images = images.to(device)
            lidars = [lidar.to(device) for lidar in lidars]

            # Pad class_ids to a fixed length and get the primary class ID
            max_labels = 10
            padded_class_ids = [
                cls_id.tolist() + [0] * (max_labels - len(cls_id)) if len(cls_id) < max_labels else cls_id.tolist()[:max_labels]
                for cls_id in class_ids
            ]
            primary_class_ids = torch.tensor([cls_id[0] for cls_id in padded_class_ids]).to(device)

            # Forward pass
            outputs = model(images, lidars)
            
            # Compute loss (using primary class IDs for the criterion)
            loss = criterion(outputs, primary_class_ids)
            
            # Accumulate loss and calculate accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += primary_class_ids.size(0)
            correct += predicted.eq(primary_class_ids).sum().item()
            # Increment step count
            step_count += 1
            # print(f"Step {step_count}/{max_steps} completed")

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy