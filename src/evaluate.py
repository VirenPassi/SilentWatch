import torch
import numpy as np
from models.model import SilentWatchModel
import os
import yaml
from torch.utils.data import DataLoader

def evaluate_model(model_path, data_loader, device, num_classes=14, tolerance=5):
    """
    Evaluates the model on classification accuracy and Time-of-Action (ToA) regression.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    model = SilentWatchModel(
        num_classes=num_classes,
        temporal_module=config['model']['temporal_module'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    all_cls_preds = []
    all_cls_labels = []
    all_toa_preds = []
    all_toa_labels = []
    
    with torch.no_grad():
        for frames, labels, toas in data_loader:
            frames, labels, toas = frames.to(device), labels.to(device), toas.to(device)
            
            cls_out, toa_out = model(frames)
            
            # Classification
            _, cls_preds = torch.max(cls_out, 1)
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_labels.extend(labels.cpu().numpy())
            
            # ToA Regression
            all_toa_preds.extend(toa_out.cpu().numpy())
            all_toa_labels.extend(toas.cpu().numpy())
            
    # Calculate Metrics
    all_cls_preds = np.array(all_cls_preds)
    all_cls_labels = np.array(all_cls_labels)
    all_toa_preds = np.array(all_toa_preds)
    all_toa_labels = np.array(all_toa_labels)
    if not np.isfinite(all_toa_labels).all():
        raise ValueError("Evaluation requires valid ToA annotations; found missing or non-finite labels.")
    
    # 1. Classification Accuracy and macro F1 (implemented without a new dependency).
    cls_acc = np.mean(all_cls_preds == all_cls_labels) * 100
    f1_scores = []
    for class_id in range(num_classes):
        true_positive = np.sum((all_cls_preds == class_id) & (all_cls_labels == class_id))
        false_positive = np.sum((all_cls_preds == class_id) & (all_cls_labels != class_id))
        false_negative = np.sum((all_cls_preds != class_id) & (all_cls_labels == class_id))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0)
    macro_f1 = float(np.mean(f1_scores))
    
    # 2. ToA Mean Absolute Error (MAE)
    toa_mae = np.mean(np.abs(all_toa_preds - all_toa_labels))
    
    # 3. Localization Accuracy (within tolerance)
    # within +/- tolerance frames of the ground truth.
    diff = np.abs(all_toa_preds - all_toa_labels)
    loc_acc = np.mean(diff <= tolerance) * 100
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Classification Accuracy: {cls_acc:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ToA Mean Absolute Error (MAE): {toa_mae:.4f} frames")
    print(f"Localization Accuracy (±{tolerance} frames): {loc_acc:.2f}%")
    print("-" * 30)
    
    return cls_acc, macro_f1, toa_mae, loc_acc

if __name__ == "__main__":
    from data_loader import get_dataloader
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SilentWatch Model")
    parser.add_argument("--model_path", type=str, default="silentwatch_best_model.pth", help="Path to trained model weights")
    parser.add_argument("--metadata", type=str, default="processed/metadata.csv", help="Path to metadata CSV")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting evaluation on {device}...")

    if os.path.exists(args.metadata):
        # Load configuration to get seq_length
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        loader = get_dataloader(
            args.metadata, 
            batch_size=config['training']['batch_size'], 
            seq_length=config['dataset']['seq_length']
        )
        
        evaluate_model(
            args.model_path, 
            loader, 
            device, 
            num_classes=config['model']['num_classes'],
            tolerance=config['evaluation']['toa_tolerance']
        )
    else:
        print(f"Error: Metadata file not found at {args.metadata}. Please run preprocessing first.")
