import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from torchvision import transforms

class SPHARDataset(Dataset):
    def __init__(self, metadata_path, transform=None, seq_length=16):
        """
        Args:
            metadata_path (str): Path to metadata.csv
            transform: PyTorch transforms for frames
            seq_length (int): Number of frames in a sequence
        """
        self.df = pd.read_csv(metadata_path)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.seq_length = seq_length
        
        # Action class mapping
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        print(f"Dataset initialized with {len(self.df)} samples across {len(self.classes)} classes.")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_dir = row['frame_dir']
        class_idx = self.class_to_idx[row['class']]
        if pd.isna(row['toa_frame']):
            toa_label = float("nan")
        else:
            toa_label = float(row['toa_frame'])  # sampled-frame coordinate
        
        # Load up to seq_length frames
        # SPHAR sequence selection: We take first seq_length frames (sampling per 5 frames as per table IV optionally)
        # For simplicity, we just take the first seq_length frames.
        frame_list = []
        for i in range(self.seq_length):
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
            else:
                # Padding with zero tensor if fewer frames
                img = torch.zeros(3, 224, 224)
            frame_list.append(img)
            
        # Final shape: (seq_length, C, H, W)
        frames_tensor = torch.stack(frame_list)
        
        return frames_tensor, class_idx, torch.tensor(toa_label, dtype=torch.float32)

def get_dataloader(metadata_path, batch_size=16, seq_length=16, shuffle=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SPHARDataset(metadata_path, transform=transform, seq_length=seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test script (requires metadata.csv and frames)
    # loader = get_dataloader("processed/metadata.csv")
    pass
