import pandas as pd
import numpy as np

csv_file = "/Users/Documents/thesis/program/media/zf_processed_videos/c1_38/detected_humans_annoted.csv"
# csv_file = "/Users/Documents/thesis/program/media/zf_processed_videos/RefCam_FC/detected_humans_annoted.csv"
# Read CSV file; treat empty strings as NaN.
df = pd.read_csv(csv_file, na_values=[''])

# Ensure that 'has_face' column is numeric (fill missing with 0)
df['has_face'] = pd.to_numeric(df['has_face'], errors='coerce').fillna(0)

# List of models and their corresponding columns
models = {
    'mtcnn': {'pt': 'mtcnn_pt', 'res': 'mtcnn_res'},
    'retinaFace': {'pt': 'retinaFace_pt', 'res': 'retinaFace_res'},
    'yolov8': {'pt': 'yolov8_pt', 'res': 'yolov8_res'},
    'yolov11': {'pt': 'yolov11_pt', 'res': 'yolov11_res'},
    'opencv': {'pt': 'opencv_pt', 'res': 'opencv_res'}
}

# Compute total number of unique frames (each row is a detection, but frames may repeat)
total_frames = df['frame'].nunique()

print(f"Total Frames in video: {total_frames}\n")
# Option 1: If your column is binary (0/1), you can simply sum it:
total_humans = df['is_person'].sum()
print("Total number of human detections (is_person = 1):", total_humans)
for model, cols in models.items():
    pt_col = cols['pt']
    res_col = cols['res']
    
    # Convert processing time column to numeric (ignore non-numeric rows)
    df[pt_col] = pd.to_numeric(df[pt_col], errors='coerce')
    processing_times = df[pt_col].dropna()
    
    # Filter out processing times that are exactly zero for calculating the fastest time
    nonzero_times = processing_times[processing_times > 0]
    
    if not nonzero_times.empty:
        fastest_time = nonzero_times.min()
    else:
        fastest_time = 0.0

    # Calculate remaining timing statistics using all valid processing times
    slowest_time = processing_times.max() if not processing_times.empty else 0.0
    total_time = processing_times.sum() if not processing_times.empty else 0.0
    average_time = total_time / total_frames if total_frames > 0 else 0.0

    # Convert detection results to numeric (assume missing values mean 0)
    df[res_col] = pd.to_numeric(df[res_col], errors='coerce').fillna(0)

    # Calculate face detection statistics:
    total_faces_detected = (df[res_col] == 1).sum()
    accurate = ((df[res_col] == 1) & (df['has_face'] == 1)).sum()
    wrongly_detected = ((df[res_col] == 1) & (df['has_face'] == 0)).sum()
    missed_face = ((df[res_col] == 0) & (df['has_face'] == 1)).sum()
    
    # Print the statistics in the desired text format.
    print(f"Model: {model}")
    print(f"Fastest Time: {fastest_time:.4f}s, Slowest Time: {slowest_time:.4f}s, Total Time: {total_time:.4f}s, Total Frames: {total_frames}, Average Time per Frame: {average_time:.4f}s")
    print(f"Total faces detected: {total_faces_detected}, accurate detected face: {accurate}, wrongly detected: {wrongly_detected}, missed face: {missed_face}")
    print("-" * 80)