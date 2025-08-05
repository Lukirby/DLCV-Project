#!/usr/bin/env python3
"""
Domain Adaptation with Maximum Mean Discrepancy (MMD) for Semantic Segmentation
Converting notebook functionality to a standalone Python script.
"""

import os
import ssl
import time
import random
import gc
from contextlib import contextmanager
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import deeplabv3_resnet50

from cityscapesscripts.helpers import labels


# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, targets_dir, image_transform=None, target_transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            targets_dir (string): Directory with all the target masks.
            image_transform (callable, optional): Optional transform to be applied on images.
            target_transform (callable, optional): Optional transform to be applied on targets.
        """
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.image_transform = image_transform
        self.target_transform = target_transform
        
        # Get all image filenames
        self.image_filenames = [f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.png'))]
        self.image_filenames.sort()
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load target mask
        target_name = img_name.replace('.png', '_trainId.png')
        target_path = os.path.join(self.targets_dir, target_name)
        target = Image.open(target_path)
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.1, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - logits (non softmaxati)
        targets: (N, H, W)   - ground truth con classi (0...C-1)
        """
        num_classes = inputs.shape[1]
        device = inputs.device  # Get device from input tensor
        
        # Softmax sulle predizioni
        probs = F.softmax(inputs, dim=1)  # (N, C, H, W)
        
        # Handle ignore_index by creating a mask and filtering out ignored pixels
        if self.ignore_index is not None:
            # Create mask for valid pixels
            valid_mask = (targets != self.ignore_index)  # (N, H, W)
            
            # Only process valid pixels
            valid_targets = targets[valid_mask]  # (N_valid,)
            
            # Reshape probs to match and filter valid pixels
            probs_reshaped = probs.permute(0, 2, 3, 1)  # (N, H, W, C)
            valid_probs = probs_reshaped[valid_mask]  # (N_valid, C)
            
        else:
            # No ignore_index, process all pixels
            valid_targets = targets.view(-1)  # (N*H*W,)
            valid_probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # (N*H*W, C)
        
        # One-hot encoding of valid targets only
        if len(valid_targets) > 0:
            targets_one_hot = F.one_hot(valid_targets, num_classes=num_classes).float()  # (N_valid, C)
        else:
            # If no valid pixels, create a zero loss that maintains gradients
            # Use a small operation on the input to maintain gradient flow
            zero_loss = (inputs * 0.0).sum()  # This maintains gradients from inputs
            return zero_loss
        
        # Calcolo Dice per ogni classe usando solo pixel validi
        intersection = (valid_probs * targets_one_hot).sum(dim=0)  # (C,)
        union = valid_probs.sum(dim=0) + targets_one_hot.sum(dim=0)  # (C,)
        
        smooth_tensor = torch.tensor(self.smooth, device=device, dtype=intersection.dtype)
        dice = (2.0 * intersection + smooth_tensor) / (union + smooth_tensor)
        
        # Media sulle classi
        loss = 1.0 - dice.mean()
        return loss


# Define MMD loss between source and target feature batches
def rbf_kernel(a, b, sigma):
    a = a.unsqueeze(1) 
    b = b.unsqueeze(0) 
    dist = (a - b).pow(2).sum(2) 
    return torch.exp(-dist / (2 * sigma ** 2))


def MMD_loss(x, y, sigma=1.0):
    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def MMD_loss_ignore_index(x, y, sigma=1.0, ignore_index=255):
    # Flatten if needed (e.g., for segmentation outputs)
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)

    # Create mask for ignore_index
    x_mask = ~(x_flat == ignore_index).any(dim=1)
    y_mask = ~(y_flat == ignore_index).any(dim=1)

    x_valid = x_flat[x_mask]
    y_valid = y_flat[y_mask]

    # If no valid samples, return zero loss with gradient
    if x_valid.size(0) == 0 or y_valid.size(0) == 0:
        return (x.sum() * 0.0) + (y.sum() * 0.0)  # This maintains gradients from x and y

    K_xx = rbf_kernel(x_valid, x_valid, sigma)
    K_yy = rbf_kernel(y_valid, y_valid, sigma)
    K_xy = rbf_kernel(x_valid, y_valid, sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


# Disable SSL verification for urllib requests on MacOS
@contextmanager
def no_ssl_verification():
    """Temporarily disable SSL verification"""
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = old_context


def cityscapes_mask_to_train_ids(mask):
    mask_np = np.array(mask)
    mask_np = np.where(mask_np <= 18, mask_np, 255).astype(np.uint8)
    return torch.from_numpy(mask_np).long()


def compute_iou(preds, labels, num_classes, ignore_index=255):
    preds = torch.argmax(preds, dim=1).detach().cpu()  # [B, H, W]
    labels = labels.detach().cpu() 
    
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        
        # Escludi pixel ignorati
        mask = (labels != ignore_index)
        pred_inds = pred_inds & mask
        target_inds = target_inds & mask

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            continue  # salta classe non presente
        ious.append(intersection / union)
    
    if len(ious) == 0:
        return float('nan')  # o 0.0 se preferisci
    return sum(ious) / len(ious)


def clear_memory():
    """Clear memory and cache for all device types"""
    # Run garbage collector
    gc.collect()
    
    # Get device from global scope
    if 'DEVICE' in globals():
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        if DEVICE.type == 'mps':
            torch.mps.empty_cache()

    # Second GC run to make sure everything is cleaned up
    gc.collect()


def create_criterion(ce_importance=0.7):
    """Create the combined loss criterion"""
    CE_loss = nn.CrossEntropyLoss(ignore_index=255)  # 255 = unlabeled
    Dice_loss = DiceLoss(smooth=0.1, ignore_index=255)  

    def criterion(outputs, targets):
        ce_loss = CE_loss(outputs, targets)
        dice_loss = Dice_loss(outputs, targets)
        cls_loss = ce_importance * ce_loss + (1 - ce_importance) * dice_loss
        return cls_loss
    
    return criterion


def train_and_validate_epoch(
        model, optimizer, criterion, 
        syn_train_dataloader, syn_val_dataloader,
        real_train_dataloader, real_val_dataloader,
        mmd_weight=0.1, num_classes=19, device=None
):
    if device is None:
        device = next(model.parameters()).device

    # Training phase
    model.train()
    train_total_loss = 0
    train_cls_loss = 0
    train_mmd_loss = 0
    train_iou_source = 0
    train_iou_target = 0

    syn_train_iter = iter(syn_train_dataloader)
    real_train_iter = iter(real_train_dataloader)
    
    steps = min(len(syn_train_iter), len(real_train_iter))

    for i in tqdm(range(steps), desc="Training"):
        source_data, source_labels = next(syn_train_iter)
        target_data, target_labels = next(real_train_iter)

        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data = target_data.to(device)

        optimizer.zero_grad()  

        source_backbone_out = model.backbone(source_data)
        source_embeddings = source_backbone_out['out']

        with torch.no_grad():
            target_backbone_out = model.backbone(target_data)
            target_embeddings = target_backbone_out['out']

        target_embeddings.requires_grad = True

        source_output = model(source_data)['out']
        target_output = model(target_data)['out']

        cls_loss = criterion(source_output, source_labels)
        mmd_loss = MMD_loss_ignore_index(source_embeddings, target_embeddings)

        loss = cls_loss + mmd_weight * mmd_loss
        
        loss.backward()
        optimizer.step()

        # Collect metrics (detach to prevent graph retention)
        train_total_loss += loss.detach().cpu().item()
        train_cls_loss += cls_loss.detach().cpu().item()
        train_mmd_loss += mmd_loss.detach().cpu().item()

        iou_target = compute_iou(target_output.detach().cpu(), target_labels, num_classes)
        iou_source = compute_iou(source_output.detach().cpu(), source_labels, num_classes)
        
        train_iou_source += iou_source
        train_iou_target += iou_target
        
        # Explicit cleanup of tensors to prevent memory leaks
        del source_data, source_labels, target_data, target_labels
        del source_embeddings, target_embeddings, source_output, target_output
        del loss, cls_loss, mmd_loss
        
        # Periodically clear memory (every 5 batches)
        if (i + 1) % 5 == 0:
            clear_memory()  
    
    clear_memory()
    
    source_train_avg_loss = train_total_loss / steps
    source_train_avg_iou = train_iou_source / steps
    target_train_avg_iou = train_iou_target / steps

    train_avg_mmd_loss = train_mmd_loss / steps
    
    # Validation 
    model.eval()
    val_total_loss = 0
    val_iou_source = 0
    val_iou_target = 0

    val_steps = min(len(syn_val_dataloader), len(real_val_dataloader))
    
    with torch.no_grad():
        for i in tqdm(range(val_steps), desc="Validation"):
            source_data, source_labels = next(iter(syn_val_dataloader))
            target_data, target_labels = next(iter(real_val_dataloader))
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data, target_labels = target_data.to(device), target_labels.to(device)

            outputs_source = model(source_data)['out']
            outputs_target = model(target_data)['out']

            loss_real = criterion(outputs_source, source_labels)
            loss_target = criterion(outputs_target, target_labels)

            val_total_loss += loss_real.detach().cpu().item() 
            val_iou_source += compute_iou(outputs_source.detach().cpu(), source_labels, num_classes)
            val_iou_target += compute_iou(outputs_target.detach().cpu(), target_labels, num_classes)

            # Explicit cleanup
            del source_data, source_labels, target_data, target_labels
            del outputs_source, outputs_target, loss_real, loss_target
            
            # Periodic memory clear
            if (i + 1) % 5 == 0:
                clear_memory()
    
    clear_memory()
    
    # Average losses and IOUs
    source_val_avg_loss = val_total_loss / val_steps
    source_val_avg_iou = val_iou_source / val_steps
    target_val_avg_iou = val_iou_target / val_steps

    print(f"The average mmd loss is {train_avg_mmd_loss:.4f}")
    return source_train_avg_loss, source_train_avg_iou, target_train_avg_iou, \
            source_val_avg_loss, source_val_avg_iou, target_val_avg_iou


def domain_adaptation(
        start_epoch, end_epoch,
        model, name,
        optimizer, criterion, 
        syn_train_dataloader, syn_val_dataloader, 
        real_train_dataloader, real_val_dataloader,
        seed=42, num_classes=19, mmd_weight=0.1, lr=1e-4
):
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Store metrics for plotting
    st_losses = []  # Source Training losses
    st_ious = []  # Source Training IoUs
    tt_losses = []  # Target Training losses
    sv_losses = []  # Source Validation losses
    sv_ious = []  # Source Validation IoUs
    tv_ious = []  # Target Validation IoUs

    # Track best validation IoU for saving best model
    best_source_val_iou = 0.0
    best_target_val_iou = 0.0
    
    # Check if we need to resume from a checkpoint
    resume_epoch = start_epoch
    
    if start_epoch > 0:
        # Load checkpoint from the specified start_epoch
        checkpoint_path = f"models/{name}_epoch_{start_epoch}.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Load model and optimizer state
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load metrics history
            st_losses = checkpoint.get("st_losses", [])
            st_ious = checkpoint.get("st_ious", [])
            tt_losses = checkpoint.get("tt_losses", [])
            sv_losses = checkpoint.get("sv_losses", [])
            sv_ious = checkpoint.get("sv_ious", [])
            tv_ious = checkpoint.get("tv_ious", [])
            
            print(f"Resumed from epoch {start_epoch}")
            print(f"Last metrics - Source Val IoU: {sv_ious[-1] if sv_ious else 'N/A':.4f}, Target Val IoU: {tv_ious[-1] if tv_ious else 'N/A':.4f}")
            
            del checkpoint
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting training from scratch.")
    else:
        print(f"Starting training from scratch (start_epoch = 0).")
    
    # Load best validation IoUs from best model checkpoints
    best_source_checkpoint_path = f"models/{name}_best_unsupervised_model.pth"
    best_target_checkpoint_path = f"models/{name}_best_supervised_model.pth"
    
    if os.path.exists(best_source_checkpoint_path):
        best_source_checkpoint = torch.load(best_source_checkpoint_path, map_location=DEVICE)
        best_source_val_iou = best_source_checkpoint.get("source_val_avg_iou", 0.0)
        print(f"Loaded best source validation IoU: {best_source_val_iou:.4f}")
        del best_source_checkpoint
    
    if os.path.exists(best_target_checkpoint_path):
        best_target_checkpoint = torch.load(best_target_checkpoint_path, map_location=DEVICE)
        best_target_val_iou = best_target_checkpoint.get("target_val_avg_iou", 0.0)
        print(f"Loaded best target validation IoU: {best_target_val_iou:.4f}")
        del best_target_checkpoint
    
    start = time.time()
    set_seeds(seed+resume_epoch)
    
    for epoch in range(resume_epoch, end_epoch):
        print(f"Epoch {epoch+1}/{end_epoch}")
        
        # Train and validate in one go
        source_train_avg_loss, source_train_avg_iou, target_train_avg_iou, \
            source_val_avg_loss, source_val_avg_iou, target_val_avg_iou = \
        train_and_validate_epoch(
            model, optimizer, criterion, 
            syn_train_dataloader, syn_val_dataloader,
            real_train_dataloader, real_val_dataloader,
            mmd_weight=mmd_weight, num_classes=num_classes
        )
        
        # Store metrics
        st_losses.append(source_train_avg_loss)
        st_ious.append(source_train_avg_iou)
        tt_losses.append(target_train_avg_iou)
        sv_losses.append(source_val_avg_loss)
        sv_ious.append(source_val_avg_iou)
        tv_ious.append(target_val_avg_iou)
        
        # Print results
        print(f"Source Train Loss: {source_train_avg_loss:.4f} | Source Train IoU: {source_train_avg_iou:.4f}")
        print(f"Target Train IoU: {target_train_avg_iou:.4f}")
        print(f"Source Val Loss: {source_val_avg_loss:.4f} | Source Val IoU: {source_val_avg_iou:.4f}")
        print(f"Target Val IoU: {target_val_avg_iou:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "source_train_avg_loss": source_train_avg_loss,
            "source_train_avg_iou": source_train_avg_iou,
            "target_train_avg_iou": target_train_avg_iou,
            "source_val_avg_loss": source_val_avg_loss,
            "source_val_avg_iou": source_val_avg_iou,
            "target_val_avg_iou": target_val_avg_iou,
            "st_losses": st_losses,
            "st_ious": st_ious,
            "tt_losses": tt_losses,
            "sv_losses": sv_losses,
            "sv_ious": sv_ious,
            "tv_ious": tv_ious,
            "num_classes": num_classes,
            "learning_rate": lr,
        }

        # Save best model based on validation IoU
        source_val_iou = source_val_avg_iou
        target_val_iou = target_val_avg_iou

        if source_val_iou > best_source_val_iou:
            best_source_val_iou = source_val_iou
            best_model_path = f"models/{name}_best_unsupervised_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved: {best_model_path} (Source Val IoU: {source_val_iou:.4f})")

        if target_val_iou > best_target_val_iou:
            best_target_val_iou = target_val_iou
            best_model_path = f"models/{name}_best_supervised_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved: {best_model_path} (Target Val IoU: {target_val_iou:.4f})")
        
        checkpoint_path = f"models/{name}_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Delete old checkpoints if they exist
        checkpoint_delete = f"models/{name}_epoch_{epoch}.pth"
        if os.path.exists(checkpoint_delete):
            os.remove(checkpoint_delete)
            print(f"Deleted old checkpoint: {checkpoint_delete}")
        
        del checkpoint
        print("-" * 50)
        
    end = time.time()
    print(f"\nTraining completed in {end - start:.2f} seconds, with {end_epoch} epochs.")
    print(f"Best source validation IoU: {best_source_val_iou:.4f}")
    print(f"Best target validation IoU: {best_target_val_iou:.4f}")
    print(f"Final model saved as: models/{name}_epoch_{end_epoch}.pth")


def create_datasets_and_dataloaders(batch_size=4, seed=42):
    """Create datasets and dataloaders for both synthetic and real data"""
    
    # Synthetic dataset paths
    path_images = "syn_resized_images"
    path_target = "syn_resized_gt"

    # Image and target transforms
    image_transform = transforms.Compose([
        transforms.Resize((256, 466)),  # We maintain the og aspect ratio
        transforms.ToTensor(),  # Converts PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet parameters
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 466), interpolation=Image.NEAREST),  # This interpolation ensure that all pixels have a correct value of their class
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])

    # Create synthetic dataset
    syn_dataset = SegmentationDataset(
        images_dir=path_images, 
        targets_dir=path_target, 
        image_transform=image_transform, 
        target_transform=target_transform
    )

    # Split synthetic dataset
    total_size = len(syn_dataset)
    print(f"Total synthetic dataset size: {total_size}")

    # Calculate split sizes (60% train, 10% val, 30% test)
    train_size = int(0.6 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Synthetic Train size: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"Synthetic Validation size: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"Synthetic Test size: {test_size} ({test_size/total_size*100:.1f}%)")

    # Create random splits
    syn_train_dataset, syn_val_dataset, syn_test_dataset = random_split(
        syn_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)  # For reproducibility
    )

    # Create DataLoaders for synthetic data
    syn_train_dataloader = DataLoader(
        syn_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed) 
    )

    syn_val_dataloader = DataLoader(
        syn_val_dataset,
        batch_size=batch_size,
        shuffle=True,  
        generator=torch.Generator().manual_seed(seed)
    )

    syn_test_dataloader = DataLoader(
        syn_test_dataset,
        batch_size=batch_size,
        shuffle=True,  
        generator=torch.Generator().manual_seed(seed)
    )

    # Real dataset (Cityscapes)
    real_target_transform = transforms.Compose([
        transforms.Resize((256, 466), interpolation=Image.NEAREST),  # This interpolation ensure that all pixels have a correct value of their class
        transforms.Lambda(lambda x: cityscapes_mask_to_train_ids(x))  # Convert Cityscapes mask to train IDs
    ])

    real_train_dataset = Cityscapes(
        root='cityscapes',
        split='train',
        mode='fine',
        target_type='semantic',
        transform=image_transform,
        target_transform=real_target_transform
    )

    real_val_dataset = Cityscapes(
        root='cityscapes',
        split='val',
        mode='fine',
        target_type='semantic',
        transform=image_transform,
        target_transform=real_target_transform
    )

    real_test_dataset = Cityscapes(
        root='cityscapes',
        split='test',
        mode='fine',
        target_type='semantic',
        transform=image_transform,
        target_transform=real_target_transform
    )

    # Print real dataset sizes
    real_total_size = len(real_train_dataset) + len(real_val_dataset) + len(real_test_dataset)
    print(f"Total real dataset size: {real_total_size}")
    print(f"Real Train size: {len(real_train_dataset)} ({len(real_train_dataset)/real_total_size*100:.1f}%)")
    print(f"Real Validation size: {len(real_val_dataset)} ({len(real_val_dataset)/real_total_size*100:.1f}%)")
    print(f"Real Test size: {len(real_test_dataset)} ({len(real_test_dataset)/real_total_size*100:.1f}%)")

    # Create DataLoaders for real data
    real_train_dataloader = DataLoader(
        real_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    real_val_dataloader = DataLoader(
        real_val_dataset,
        batch_size=batch_size,
        shuffle=True,  
        generator=torch.Generator().manual_seed(seed)
    )

    real_test_dataloader = DataLoader(
        real_test_dataset,
        batch_size=batch_size,
        shuffle=True,  
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"\nDataLoaders created:")
    print(f"Synthetic - Train batches: {len(syn_train_dataloader)}, Val batches: {len(syn_val_dataloader)}, Test batches: {len(syn_test_dataloader)}")
    print(f"Real - Train batches: {len(real_train_dataloader)}, Val batches: {len(real_val_dataloader)}, Test batches: {len(real_test_dataloader)}")

    return (syn_train_dataloader, syn_val_dataloader, syn_test_dataloader,
            real_train_dataloader, real_val_dataloader, real_test_dataloader)


def create_model(num_classes=19, device=None, pretrained_path=None):
    """Create and load the DeepLabV3 model"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() 
                              else "mps" if torch.mps.is_available() 
                              else "cpu")
    
    # Load model
    if device.type == 'mps': 
        print("mps detected, using no_ssl_verification")
        with no_ssl_verification():
            model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    else:
        model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.to(device)

    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier.")}
        model.load_state_dict(filtered_state_dict, strict=False)
    
    return model


def main():
    """Main function to run domain adaptation training"""
    
    # Configuration
    SEED = 42
    NUM_CLASSES = 19
    LR = 1e-4
    BATCH_SIZE = 4
    start_epoch = 5
    end_epochs = 10  # Epochs to end train
    MMD_WEIGHT = 0.2
    CE_IMPORTANCE = 0.7
    
    # Set device
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.mps.is_available() 
                          else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Set seeds for reproducibility
    set_seeds(SEED)
    print(f"Seeds set for reproducibility: {SEED}")
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    (syn_train_dataloader, syn_val_dataloader, syn_test_dataloader,
     real_train_dataloader, real_val_dataloader, real_test_dataloader) = create_datasets_and_dataloaders(
        batch_size=BATCH_SIZE, seed=SEED
    )
    
    # Create model
    print("Creating model...")
    # Set to None to start fresh, or specify a path to load a specific model
    best_model_path = "models/deeplabv3_imagenet1k_best_model.pth"

    # Model name for saving
    name = "DA_highermmd_deeplabv3_imagenet1k"

    model = create_model(
        num_classes=NUM_CLASSES, 
        device=DEVICE, 
        pretrained_path=best_model_path if best_model_path and os.path.exists(best_model_path) else None
    )
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = create_criterion(ce_importance=CE_IMPORTANCE)
    
    
    # Run domain adaptation
    print("Starting domain adaptation training...")
    domain_adaptation(
        start_epoch=start_epoch,  
        end_epoch=end_epochs,  
        model=model,
        name=name, 
        optimizer=optimizer, 
        criterion=criterion, 
        syn_train_dataloader=syn_train_dataloader, 
        syn_val_dataloader=syn_val_dataloader, 
        real_train_dataloader=real_train_dataloader, 
        real_val_dataloader=real_val_dataloader,
        seed=SEED,
        num_classes=NUM_CLASSES,
        mmd_weight=MMD_WEIGHT,
        lr=LR
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()