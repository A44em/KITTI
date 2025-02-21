# Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import kitti_hndlr
import radarscenes_hndlr
import fusion_hndlr

from sklearn.metrics import precision_recall_curve, average_precision_score

EPOCHS = 50

def compute_classwise_ap(predictions, labels, num_classes):
    """
    Compute Average Precision (AP) for each class.

    Args:
        predictions (torch.Tensor): Model predictions of shape (num_samples, num_classes).
        labels (torch.Tensor): Ground-truth labels of shape (num_samples,).
        num_classes (int): Total number of classes.

    Returns:
        dict: A dictionary with AP for each class and the mean mAP.
    """
    ap_per_class = {}
    for class_idx in range(num_classes):
        # Convert to binary labels for this class
        binary_labels = (labels == class_idx).float().cpu().numpy()
        class_predictions = predictions[:, class_idx].detach().cpu().numpy()

        # Compute precision-recall curve and AP
        ap = average_precision_score(binary_labels, class_predictions)
        ap_per_class[class_idx] = ap

    # Compute mean Average Precision (mAP)
    mAP = sum(ap_per_class.values()) / num_classes
    ap_per_class["mAP"] = mAP

    return ap_per_class

def collate_fn(batch):
    # Unpack the batch into respective components
    images, lidars, class_ids, bboxes, dimensions, locations = zip(*batch)

    # Stack images and lidars, which have uniform shapes
    images = torch.stack(images)
    lidars = list(lidars)

    # Class IDs, bounding boxes, dimensions, and locations may have variable length due to multiple objects per image.
    # So, we need to keep them as lists (or pad them if needed for a specific model).
    class_ids = [torch.tensor(ids) for ids in class_ids]
    bboxes = [torch.tensor(boxes) for boxes in bboxes]
    dimensions = [torch.tensor(dims) for dims in dimensions]
    locations = [torch.tensor(locs) for locs in locations]

    return images, lidars, class_ids, bboxes, dimensions, locations

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download datasets
    url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
    kitti_hndlr.download_kitti_dataset_img(url, '/home/a44em_3li/KITTI_img')
    url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip'
    kitti_hndlr.download_kitti_dataset_lidar(url, '/home/a44em_3li/KITTI_lidar')
    url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
    kitti_hndlr.download_kitti_dataset_labels(url, '/home/a44em_3li/KITTI_labels')
    url = 'https://zenodo.org/records/4559821/files/RadarScenes.zip'
    radarscenes_hndlr.download_radarscenes_dataset(url, '/home/a44em_3li/RadarScenes')

    # Load datasets and split into training/validation sets
    image_dir = '/home/a44em_3li/KITTI_img/training/image_2'
    lidar_dir = '/home/a44em_3li/KITTI_lidar/training/velodyne'
    labels_dir = '/home/a44em_3li/KITTI_labels/training/label_2'


    # Define your transform with resizing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to a uniform size
        transforms.ToTensor()
    ])
  
    kitti_dataset = kitti_hndlr.KITTIDataset(image_dir, lidar_dir, labels_dir, transform=transform, class_mapping=kitti_hndlr.class_mapping)

    radar_dataset = radarscenes_hndlr.RadarScenesDataset('/home/a44em_3li/RadarScenes/RadarScenes/data/')
    radar_dataset = radarscenes_hndlr.process_dataset(radar_dataset)

    # Split into training and validation sets (80% train, 20% validation)
    kitti_train_size = int(0.8 * len(kitti_dataset))
    kitti_val_size = len(kitti_dataset) - kitti_train_size
    kitti_train_dataset, kitti_val_dataset = random_split(kitti_dataset, [kitti_train_size, kitti_val_size])

    radar_train_size = int(0.8 * len(radar_dataset))
    radar_val_size = len(radar_dataset) - radar_train_size
    radar_train_dataset, radar_val_dataset = random_split(radar_dataset, [radar_train_size, radar_val_size])

    # Create DataLoaders
    kitti_train_loader = DataLoader(kitti_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    kitti_val_loader = DataLoader(kitti_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
    radar_train_loader = DataLoader(radar_train_dataset, batch_size=64, shuffle=True)
    radar_val_loader = DataLoader(radar_val_dataset, batch_size=64, shuffle=False)

    # Define Models
    kitti_model = kitti_hndlr.KITTICNNModel().to(device)
    radar_model = radarscenes_hndlr.RadarRNNModel().to(device)

    # Define a learnable fusion layer (this example assumes CNN and RNN output size is 12, and the final output size is 12 classes)
    fusion_model = fusion_hndlr.LearnableFusion(cnn_output_size=12, rnn_output_size=12, fusion_output_size=12).to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_kitti = optim.Adam(kitti_model.parameters(), lr=0.001)
    optimizer_radar = optim.Adam(radar_model.parameters(), lr=0.001)
    optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=0.001)

    # Load the checkpoint
    kitti_checkpoint_path = "kitti_model_checkpoint.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(kitti_checkpoint_path)  # For GPU
    else:
        checkpoint = torch.load(kitti_checkpoint_path, map_location=torch.device('cpu'))  # For CPU

    # Restore the model, optimizer, and other states
    kitti_model.load_state_dict(checkpoint['kitti_model_state_dict'])
    optimizer_kitti.load_state_dict(checkpoint['optimizer_kitti_state_dict'])

    # Train and Validate KITTI model
    print("Training KITTI CNN Model...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = kitti_hndlr.train_model(kitti_model, kitti_train_loader, criterion, optimizer_kitti, device)
        val_loss, val_acc = kitti_hndlr.validate_model(kitti_model, kitti_val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save the trained KITTI model
    # torch.save(kitti_model.state_dict(), "kitti_model.pth")
    # print("KITTI model saved to kitti_model.pth")
        torch.save({
            'kitti_model_state_dict': kitti_model.state_dict(),
            'optimizer_kitti_state_dict': optimizer_kitti.state_dict(),
        }, "kitti_model_checkpoint.pth")
        print("KITTI model saved to kitti_model_checkpoint.pth")

    # Load the checkpoint
    radar_checkpoint_path = "radar_model_checkpoint.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(radar_checkpoint_path)  # For GPU
    else:
        checkpoint = torch.load(radar_checkpoint_path, map_location=torch.device('cpu'))  # For CPU

    # Restore the model, optimizer, and other states
    radar_model.load_state_dict(checkpoint['radar_model_state_dict'])
    optimizer_radar.load_state_dict(checkpoint['optimizer_radar_state_dict'])

    # Train and Validate Radar model
    print("Training Radar RNN Model...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = radarscenes_hndlr.train_model_radar(radar_model, radar_train_loader, criterion, optimizer_radar, device)
        val_loss, val_acc = radarscenes_hndlr.validate_model_radar(radar_model, radar_val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


    # Save the trained Radar model
    # torch.save(radar_model.state_dict(), "radar_model.pth")
    # print("Radar model saved to radar_model.pth")
        torch.save({
            'radar_model_state_dict': radar_model.state_dict(),
            'optimizer_radar_state_dict': optimizer_radar.state_dict(),
        }, "radar_model_checkpoint.pth")
        print("Radar model saved to radar_model_checkpoint.pth")

    # Load the checkpoint
    fusion_checkpoint_path = "fusion_model_checkpoint.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(fusion_checkpoint_path)  # For GPU
    else:
        checkpoint = torch.load(fusion_checkpoint_path, map_location=torch.device('cpu'))  # For CPU

    # Restore the model, optimizer, and other states
    fusion_model.load_state_dict(checkpoint['fusin_model_state_dict'])
    optimizer_fusion.load_state_dict(checkpoint['optimizer_fusion_state_dict'])

    for kitti_batch, radar_batch in zip(kitti_val_loader, radar_val_loader):
        # Print the structure and shapes of the items in each batch
        print("KITTI Batch Structure:")
        for i, item in enumerate(kitti_batch):
            print(f"  Item {i}: Type: {type(item)}, Shape: {item.shape if isinstance(item, torch.Tensor) else 'Not a Tensor'}")
        
        print("\nRadar Batch Structure:")
        for i, item in enumerate(radar_batch):
            print(f"  Item {i}: Type: {type(item)}, Shape: {item.shape if isinstance(item, torch.Tensor) else 'Not a Tensor'}")
        
        # Stop after the first batch for inspection
        break
    # Perform Fusion on Validation set
    print("Performing Learnable Fusion on Validation Data...")
    fusion_model.train()
    fusion_correct = 0
    fusion_total = 0
    running_fusion_loss = 0.0

    # Initialize storage for predictions and labels
    all_kitti_predictions = []
    all_radar_predictions = []
    all_fusion_predictions = []
    all_labels = []

    for (kitti_data_batch, lidar_data_batch, *other_kitti_components), (radar_data_batch, labels) in zip(kitti_val_loader, radar_val_loader):
        # Move data to the appropriate device
        kitti_data_batch = kitti_data_batch.to(device)  # Assuming kitti_data_batch is a tensor
        lidar_data_batch = [lidar.to(device) for lidar in lidar_data_batch]  # Move each LiDAR tensor to the device
        radar_data_batch = radar_data_batch.to(device)
        labels = labels.to(device)

        # Get predictions from both models
        kitti_output = kitti_model(kitti_data_batch, lidar_data_batch)
        radar_output = radar_model(radar_data_batch)

        # Check the feature size (dimension 1) of both outputs
        kitti_feature_size = kitti_output.size(1)
        radar_feature_size = radar_output.size(1)

        # Pad the smaller tensor to match the larger one
        if kitti_feature_size > radar_feature_size:
            padding_size = kitti_feature_size - radar_feature_size
            # Pad radar_output to match kitti_output's feature size
            padded_radar_output = torch.nn.functional.pad(radar_output, (0, 0, 0, padding_size))  # Pad only along feature dimension
            fused_output = fusion_model(kitti_output, padded_radar_output)
        elif radar_feature_size > kitti_feature_size:
            padding_size = radar_feature_size - kitti_feature_size
            # Pad kitti_output to match radar_output's feature size
            padded_kitti_output = torch.nn.functional.pad(kitti_output, (0, 0, 0, padding_size))  # Pad only along feature dimension
            fused_output = fusion_model(padded_kitti_output, radar_output)
        else:
            # If feature sizes match, no padding needed
            fused_output = fusion_model(kitti_output, radar_output)

        # Determine the minimum batch size
        min_batch_size = min(fused_output.size(0), labels.size(0))

        # Trim both tensors to the minimum batch size
        fused_output = fused_output[:min_batch_size]
        labels = labels[:min_batch_size]

        # Compute loss for the fused output
        loss = criterion(fused_output, labels)
        running_fusion_loss += loss.item()

        # Get the final predictions and calculate accuracy
        _, fusion_predicted = fused_output.max(1)
        fusion_total += labels.size(0)
        fusion_correct += fusion_predicted.eq(labels).sum().item()

        # Calculate final fusion accuracy and loss
        fusion_accuracy = 100 * fusion_correct / fusion_total
        fusion_loss = running_fusion_loss / len(kitti_val_loader)

        # Trim all tensors to the minimum batch size
        kitti_output = kitti_output[:min_batch_size]
        radar_output = radar_output[:min_batch_size]
        fused_output = fused_output[:min_batch_size]
        labels = labels[:min_batch_size]

        # Store predictions and labels
        all_kitti_predictions.append(kitti_output)
        all_radar_predictions.append(radar_output)
        all_fusion_predictions.append(fused_output)
        all_labels.append(labels)

    # Concatenate all predictions and labels
    all_kitti_predictions = torch.cat(all_kitti_predictions, dim=0)
    all_radar_predictions = torch.cat(all_radar_predictions, dim=0)
    all_fusion_predictions = torch.cat(all_fusion_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Number of classes
    num_classes = all_fusion_predictions.size(1)

    # Calculate mAP for KITTI model
    kitti_ap_per_class = compute_classwise_ap(all_kitti_predictions, all_labels, num_classes)
    print("\nKITTI Model mAP:")
    for class_idx, ap in kitti_ap_per_class.items():
        if class_idx == "mAP":
            print(f"  Mean Average Precision (mAP): {ap:.4f}")
        else:
            print(f"  Class {class_idx} AP: {ap:.4f}")

    # Calculate mAP for Radar model
    radar_ap_per_class = compute_classwise_ap(all_radar_predictions, all_labels, num_classes)
    print("\nRadar Model mAP:")
    for class_idx, ap in radar_ap_per_class.items():
        if class_idx == "mAP":
            print(f"  Mean Average Precision (mAP): {ap:.4f}")
        else:
            print(f"  Class {class_idx} AP: {ap:.4f}")

    # Calculate mAP for Fusion model
    fusion_ap_per_class = compute_classwise_ap(all_fusion_predictions, all_labels, num_classes)
    print("\nFusion Model mAP:")
    for class_idx, ap in fusion_ap_per_class.items():
        if class_idx == "mAP":
            print(f"  Mean Average Precision (mAP): {ap:.4f}")
        else:
            print(f"  Class {class_idx} AP: {ap:.4f}")

    # Save the trained Fusion model
    # torch.save(fusion_model.state_dict(), "fusion_model.pth")
    # print("Fusion model saved to fusion_model.pth")
    # Save model and optimizer state
    torch.save({
        'fusin_model_state_dict': fusion_model.state_dict(),
        'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
    }, "fusion_model_checkpoint.pth")
    print("Fusion model saved to fusion_model_checkpoint.pth")

    print(f"Learnable Fusion Validation Loss: {fusion_loss:.4f}, Learnable Fusion Validation Accuracy: {fusion_accuracy:.2f}%")

