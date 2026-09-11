import os
import cv2
import pandas as pd
from tqdm import tqdm

def extract_frames(video_path, output_dir, target_fps=10, sampling_rate=5, target_size=(224, 224)):
    """
    Extracts frames from a video at a targeted FPS and resizes them.
    Returns: List of extracted frame paths and the total duration.
    """
    video_name = os.path.basename(video_path).split('.')[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Error message is already printed by OpenCV for some cases
            return [], None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return [], None
            
        # The paper samples one frame in every five source frames (50 -> 10 FPS).
        skip = int(sampling_rate)
        
        count = 0
        saved_count = 0
        frame_paths = []
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            if count % skip == 0:
                frame_resized = cv2.resize(frame, target_size)
                frame_path = os.path.join(frame_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame_resized)
                frame_paths.append(frame_path)
                saved_count += 1
                
            count += 1
            
        cap.release()
        return frame_paths, fps
    except Exception as e:
        print(f"\nSkipping corrupted video {video_path}: {e}")
        return [], None

def process_video(video_file, class_path, frames_base_dir, target_fps, sampling_rate, annotations):
    if not video_file.endswith(('.mp4', '.avi', '.mov')):
        return None
        
    video_path = os.path.join(class_path, video_file)
    frame_paths, source_fps = extract_frames(
        video_path, frames_base_dir, target_fps=target_fps,
        sampling_rate=sampling_rate, target_size=(224, 224)
    )
    
    if frame_paths:
        return {
            "video_name": video_file,
            "class": os.path.basename(class_path),
            "frame_count": len(frame_paths),
            "frame_dir": os.path.dirname(frame_paths[0]),
            "source_fps": source_fps,
            "sampled_fps": target_fps,
            "sampling_rate": sampling_rate,
            "frame_width": 224,
            "frame_height": 224,
            # Keep missing labels missing; training validates them explicitly.
            "toa_frame": annotations.get(video_file)
        }
    return None

def preprocess_dataset(raw_dir, processed_dir, target_fps=10, sampling_rate=5,
                       annotation_csv=None):
    os.makedirs(processed_dir, exist_ok=True)
    frames_base_dir = os.path.join(processed_dir, "extracted_frames")
    os.makedirs(frames_base_dir, exist_ok=True)
    
    annotations = {}
    if annotation_csv:
        if not os.path.exists(annotation_csv):
            raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")
        annotation_df = pd.read_csv(annotation_csv)
        if "video_name" not in annotation_df.columns:
            raise ValueError("Annotation CSV must contain a 'video_name' column")
        if "toa_frame" in annotation_df.columns:
            values = annotation_df["toa_frame"]
        elif "toa_seconds" in annotation_df.columns:
            values = annotation_df["toa_seconds"] * target_fps
        else:
            raise ValueError("Annotation CSV must contain 'toa_frame' or 'toa_seconds'")
        for name, value in zip(annotation_df["video_name"], values):
            if pd.notna(value):
                # Source-frame labels are converted to the sampled-frame coordinate.
                annotations[str(name)] = float(value) / sampling_rate

    data_list = []
    
    # Get all video files
    video_tasks = []
    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_tasks.append((video_file, class_path))

    print(f"Preprocessing {len(video_tasks)} videos using serial loop. Skipping already processed...")
    for video_file, class_path in tqdm(video_tasks):
        result = process_video(video_file, class_path, frames_base_dir, target_fps,
                               sampling_rate, annotations)
        if result:
            data_list.append(result)
    
    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(processed_dir, "metadata.csv"), index=False)
    print(f"Preprocessing complete. Metadata saved to {os.path.join(processed_dir, 'metadata.csv')}")

if __name__ == "__main__":
    # This will be run after the dataset is downloaded
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    preprocess_dataset(
        config['dataset']['raw_dir'], config['dataset']['processed_dir'],
        config['dataset']['fps'], config['dataset'].get('sampling_rate', 5),
        config['dataset'].get('annotation_csv')
    )
    print("Preprocessing complete.")
