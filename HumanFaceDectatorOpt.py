import os
from pathlib import Path
import pandas as pd
import time
from datetime import timedelta
import cv2
import torch
from ultralytics import YOLO
from retinaface import RetinaFace
from facenet_pytorch import MTCNN
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class HumanFaceDetectorOpt:

    def __init__(self, source_video_directory, processed_video_directory, models_directory, allowed_video_extensions):
        self.face_detection_models = ["mtcnn", "retinaFace", "yolov8", "yolov11", "opencv"]
        self.source_video_directory = source_video_directory
        self.processed_video_directory = processed_video_directory
        self.models_directory = models_directory
        self.video_extensions = allowed_video_extensions
        self.yolo_human_model_path = f"{self.models_directory}/yolo11n.pt"
        self.yolo_face_model_path_v8 = f"{self.models_directory}/yolov8n-face.pt"
        self.yolo_face_model_path_v11 = f"{self.models_directory}/yolov11n-face.pt"
        self.opencv_face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


    def detect_humans_in_video(self, video_path, output_dir, csv_file, yolo_human_model_path): # Added model path as argument
        """Detect humans in video frames and save bounding box data to CSV."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        person_frames_dir = os.path.join(output_dir, "person_frames")
        person_marked_frames_dir = os.path.join(output_dir, "person_marked_frames")
        Path(person_frames_dir).mkdir(parents=True, exist_ok=True)
        Path(person_marked_frames_dir).mkdir(parents=True, exist_ok=True)

        df_columns = ["frame", "x1", "y1", "x2", "y2", "image_path", "timestamp_in_video",
                      "is_person", "has_face", "mtcnn_pt", "mtcnn_res", "retinaFace_pt", "retinaFace_res",
                      "yolov8_pt", "yolov8_res", "yolov11_pt", "yolov11_res", "opencv_pt", "opencv_res"]
        df = pd.DataFrame(columns=df_columns)
        df.index.name = 'id'
        df.index = df.index + 1
        frame_count = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return None

        # Load YOLO human model here, inside the process
        yolo_human_model = YOLO(yolo_human_model_path) # Model loaded in each process

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = str(timedelta(milliseconds=timestamp_ms))

            results = yolo_human_model.predict(frame, show=False)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            human_detected = False
            frame_path = None
            marked_frame = frame.copy()

            if boxes:
                human_detected = True
                frame_path = os.path.join(person_frames_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

                for box, cls in zip(boxes, clss):
                    if yolo_human_model.names[int(cls)] == "person":
                        x1, y1, x2, y2 = map(int, box)
                        label = f"Human {len(df) + 1}"
                        cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(marked_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        new_row = {
                            "frame": frame_count, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "image_path": frame_path, "timestamp_in_video": timestamp, "is_person": True
                        }
                        df.loc[len(df) + 1] = new_row

            if human_detected:
                marked_human_path = os.path.join(person_marked_frames_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(marked_human_path, marked_frame)

            frame_count += 1

        cap.release()
        df.to_csv(csv_file, index=True)
        return csv_file


    def mtcnn_process_frame_faces(self, image, group, output_dir, frame_number, mtcnn_model_): # Added mtcnn_model_ as argument
        """Process faces in a single frame using MTCNN."""
        start_time = time.time()
        faces_found = False
        updates = []

        # Load MTCNN model here, inside the process
        mtcnn = mtcnn_model_ #MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu') # Model loaded in each process

        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    person_region_rgb = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
                    face_boxes, _ = mtcnn.detect(person_region_rgb) # Use process-local model

                    if face_boxes is not None and len(face_boxes) > 0:
                        faces_found = True
                        face_found_in_person = True
                        for fbox in face_boxes:
                            x1, y1, x2, y2 = map(int, fbox)
                            cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
            updates.append({"mtcnn_pt": round(time.time() - processing_time, 4), "mtcnn_res": int(face_found_in_person), "id": row.id})

        except Exception as e:
            print(f"Error processing frame {frame_number} with MTCNN: {e}")

        output_path = None
        if faces_found:
            output_path = os.path.join(output_dir, f"marked_frame_{frame_number}.jpg")
            cv2.imwrite(output_path, image)

        return time.time() - start_time, output_path, updates


    def retinaFace_process_frame_faces(self, image, group, output_dir, frame_number):
        """Process faces in person regions using RetinaFace."""
        start_time = time.perf_counter()
        faces_found = False
        updates = []

        # RetinaFace models are loaded dynamically within RetinaFace.detect_faces, no need to load here explicitly

        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    faces = RetinaFace.detect_faces(person_region) # RetinaFace model is loaded when detect_faces is called

                    if isinstance(faces, dict) and len(faces) > 0:
                        faces_found = True
                        face_found_in_person = True
                        for _, face_data in faces.items():
                            facial_area = face_data["facial_area"]
                            x1, y1, x2, y2 = map(int, facial_area)
                            cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
            updates.append({"retinaFace_pt": round(time.time() - processing_time, 4), "retinaFace_res": int(face_found_in_person), "id": row.id})

        except Exception as e:
            print(f"Error processing frame {frame_number} with RetinaFace: {e}")

        output_path = None
        if faces_found:
            output_path = os.path.join(output_dir, f"marked_frame_{frame_number}.jpg")
            cv2.imwrite(output_path, image)

        return time.perf_counter() - start_time, output_path, updates


    def yolo_process_frame_faces(self, image, group, output_dir, frame_number, model_type, yolo_face_model_path_v8, yolo_face_model_path_v11, confidence_threshold=0.25, imgsz=640): # Added model paths as arguments
        """Process faces using YOLOv8 or YOLOv11 face detection model."""
        start_time = time.time()
        faces_found = False
        updates = []

        # Load YOLO face model here, inside the process
        if model_type == "yolov8":
            face_model = YOLO(yolo_face_model_path_v8) # Model loaded in each process
        elif model_type == "yolov11":
            face_model = YOLO(yolo_face_model_path_v11) # Model loaded in each process
        else:
            raise ValueError(f"Unsupported YOLO model type: {model_type}")


        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    results = face_model(person_region, imgsz=imgsz, conf=confidence_threshold, verbose=False)
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            faces_found = True
                            face_found_in_person = True
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({f"{model_type}_pt": round(time.time() - processing_time, 4), f"{model_type}_res": int(face_found_in_person), "id": row.id})

        except Exception as e:
            print(f"Error processing frame {frame_number} with {model_type}: {e}")

        output_path = None
        if faces_found:
            output_path = os.path.join(output_dir, f"marked_frame_{frame_number}.jpg")
            cv2.imwrite(output_path, image)

        return time.time() - start_time, output_path, updates


    def opencv_process_frame_faces(self, image, group, output_dir, frame_number, opencv_face_cascade_path): # Added cascade path as argument
        """Process faces using OpenCV Haar Cascade."""
        start_time = time.time()
        faces_found = False
        updates = []

        # Load Haar Cascade classifier here, inside the process
        face_cascade = cv2.CascadeClassifier(opencv_face_cascade_path) # Cascade loaded in each process


        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    person_region_gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale( # Use process-local cascade
                        person_region_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                    if len(faces) > 0:
                        faces_found = True
                        face_found_in_person = True
                        for (fx, fy, fw, fh) in faces:
                            cv2.rectangle(person_region, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({"opencv_pt": round(time.time() - processing_time, 4), "opencv_res": int(face_found_in_person), "id": row.id})

        except Exception as e:
            print(f"Error processing frame {frame_number} with OpenCV: {e}")

        output_path = None
        if faces_found:
            output_path = os.path.join(output_dir, f"marked_frame_{frame_number}.jpg")
            cv2.imwrite(output_path, image)

        return time.time() - start_time, output_path, updates


    def detect_face_in_human_frames(self, csv_file, output_dir, model, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path, mtcnn_model_): # Pass model paths and mtcnn_model
        """Process frames for face detection using specified model, leveraging threading for parallelism."""
        start_time_total = time.time()
        fastest, slowest, total_time, total_images, faces_count = float('inf'), 0, 0, 0, 0
        grouped = self.read_and_group_csv(csv_file)
        total_images = len(grouped)


        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for (image_path, frame_num), group in grouped:
                if not os.path.exists(image_path):
                    continue
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image at {image_path}")
                    continue

                # Pass model paths to executor
                if model == "mtcnn":
                    future = executor.submit(self.mtcnn_process_frame_faces, image, group, output_dir, frame_num, mtcnn_model_) # Pass mtcnn_model_
                elif model == "retinaFace":
                    future = executor.submit(self.retinaFace_process_frame_faces, image, group, output_dir, frame_num)
                elif model == "opencv":
                    future = executor.submit(self.opencv_process_frame_faces, image, group, output_dir, frame_num, opencv_face_cascade_path) # Pass cascade path
                elif model in ["yolov8", "yolov11"]:
                    future = executor.submit(self.yolo_process_frame_faces, image, group, output_dir, frame_num, model, yolo_face_model_path_v8, yolo_face_model_path_v11) # Pass yolo model paths
                else:
                    print(f"Warning: Unknown face detection model: {model}")
                    continue

                futures.append(future)


            for future in as_completed(futures):
                execution_time, new_path, updates = future.result()
                if updates:
                    self.update_dataframe(updates, csv_file)
                if new_path:
                    faces_count += 1
                total_time += execution_time
                slowest = max(slowest, execution_time)
                fastest = min(fastest, execution_time)


        average_time = total_time / total_images if total_images > 0 else 0
        total_run_time = time.time() - start_time_total
        return fastest, slowest, total_time, total_images, average_time, faces_count, total_run_time


    def find_file_extensions(self, root_folder, extensions):
        """Recursively find files with specified extensions."""
        try:
            for dirpath, _, filenames in os.walk(root_folder):
                for file in filenames:
                    if file.lower().endswith(tuple(extensions)):
                        folder_name = os.path.basename(dirpath)
                        file_name_without_ext = os.path.splitext(file)[0]
                        absolute_path = os.path.join(dirpath, file)
                        yield folder_name, file_name_without_ext, absolute_path
        except Exception as e:
            print(f"Error during directory traversal: {e}")


    def create_folder_structure(self):
        """Create folder structure for processed videos."""
        try:
            for folder, segments, video_path in self.find_file_extensions(self.source_video_directory, self.video_extensions):
                save_folder = os.path.join(self.processed_video_directory, segments)
                os.makedirs(save_folder, exist_ok=True)
                os.makedirs(os.path.join(save_folder, "person_frames"), exist_ok=True)
                os.makedirs(os.path.join(save_folder, "person_marked_frames"), exist_ok=True)
                Path(os.path.join(save_folder, "info.txt")).touch(exist_ok=True)
                Path(os.path.join(save_folder, "detected_humans.csv")).touch(exist_ok=True)
                for model in self.face_detection_models:
                    os.makedirs(os.path.join(save_folder, model), exist_ok=True)
        except Exception as e:
            print(f"Error creating folder structure: {e}")


    def update_dataframe(self, updates, file_path):
        """Update DataFrame in CSV file with face detection results."""
        if not updates:
            return

        df = pd.read_csv(file_path)
        for update in updates:
            row_id = update["id"]
            for col, value in update.items():
                if col != "id":
                    df.loc[row_id -1 if row_id <= len(df) else df.index[-1], col] = value

        df.to_csv(file_path, index=False)


    def run_human_detection_on_video(self, video_path, save_folder, yolo_human_model_path): # Added model path as argument
        """Run human detection on a single video and return processing time."""
        csv_file = os.path.join(save_folder, "detected_humans.csv")
        start_time = time.time()
        self.detect_humans_in_video(video_path, save_folder, csv_file, yolo_human_model_path) # Pass model path
        total_time = time.time() - start_time
        info_text = f"Human Detection Processed time: {total_time:.4f} seconds"
        self.write_to_file(os.path.join(save_folder, "info.txt"), info_text)
        return total_time


    def run_face_detection_on_video(self, save_folder, csv_file, model, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path, mtcnn_model_): # Pass model paths and mtcnn_model
        """Run face detection on a single video's frames and return metrics."""
        start_time = time.time()
        fastest, slowest, total_time, total_images, average_time, faces_count, total_run_time = self.detect_face_in_human_frames(
            csv_file, os.path.join(save_folder, model), model, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path, mtcnn_model_) # Pass all model paths and mtcnn_model
        info_text = (f"Face Detection Details ({model}):\n"
                     f"Fastest Time: {fastest:.4f}s, Slowest Time: {slowest:.4f}s, Total Time: {total_time:.4f}s, "
                     f"Total Frames: {total_images}, Average Time per Frame: {average_time:.4f}s, Faces Count: {faces_count}, Total Run Time: {total_run_time:.4f}s")
        self.write_to_file(os.path.join(save_folder, "info.txt"), info_text)
        return total_run_time


    def process_video(self, video_tuple, yolo_human_model_path, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path): # Pass model paths
        """Process a single video for human and face detection."""
        folder, segments, video_path = video_tuple
        save_folder = os.path.join(self.processed_video_directory, segments)

        human_detection_time = self.run_human_detection_on_video(video_path, save_folder, yolo_human_model_path) # Pass human model path
        print(f"Human detection for {segments} completed in {human_detection_time:.2f} seconds.")

        csv_file = os.path.join(save_folder, "detected_humans.csv")
        face_detection_times = {}

        # Load MTCNN model here, once per process, and pass it to face detection function
        mtcnn_model_process = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu') # Load MTCNN per process

        for model in self.face_detection_models:
            face_detection_time = self.run_face_detection_on_video(save_folder, csv_file, model, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path, mtcnn_model_process) # Pass all model paths and mtcnn_model_process
            face_detection_times[model] = face_detection_time
            print(f"Face detection with {model} for {segments} completed in {face_detection_time:.2f} seconds.")

        return segments, human_detection_time, face_detection_times


    def run_parallel_processing(self):
        """Process videos in parallel using multiprocessing."""
        video_files = list(self.find_file_extensions(self.source_video_directory, self.video_extensions))
        if not video_files:
            print("No video files found to process.")
            return

        cpu_count = os.cpu_count()
        print(f"Starting parallel processing using {cpu_count} cores.")
        start_overall_time = time.time()

        # Pass model paths to process_video
        yolo_human_model_path = self.yolo_human_model_path
        yolo_face_model_path_v8 = self.yolo_face_model_path_v8
        yolo_face_model_path_v11 = self.yolo_face_model_path_v11
        opencv_face_cascade_path = self.opencv_face_cascade_path


        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(self.process_video, video, yolo_human_model_path, yolo_face_model_path_v8, yolo_face_model_path_v11, opencv_face_cascade_path) for video in video_files] # Pass model paths to process_video
            for future in as_completed(futures):
                segments, human_detection_time, face_detection_times = future.result()
                print(f"Finished processing video segment: {segments}")
                print(f"  Human Detection Time: {human_detection_time:.2f} seconds")
                for model, time_taken in face_detection_times.items():
                    print(f"  Face Detection ({model}) Time: {time_taken:.2f} seconds")

        overall_processing_time = time.time() - start_overall_time
        print(f"Overall processing completed in {overall_processing_time:.2f} seconds.")


    def read_and_group_csv(self, csv_file):
        """Reads CSV and groups by image path and frame number, handling potential errors."""
        try:
            df = pd.read_csv(csv_file)
            grouped = df.groupby(['image_path', 'frame'])
            return grouped
        except FileNotFoundError:
            print(f"Error: CSV file not found: {csv_file}")
            return pd.DataFrame().groupby(['image_path', 'frame'])
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file is empty: {csv_file}")
            return pd.DataFrame().groupby(['image_path', 'frame'])
        except Exception as e:
            print(f"Error reading or grouping CSV {csv_file}: {e}")
            return pd.DataFrame().groupby(['image_path', 'frame'])


    def write_to_file(self, file_path, text):
        """Writes text to a file, creating file if it doesn't exist and appending."""
        try:
            with open(file_path, 'a') as file:
                file.write(text + '\n')
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")

