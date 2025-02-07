
import os
from pathlib import Path
import pandas as pd
from pathlib import Path
import time
from datetime import timedelta
import cv2
import torch
from ultralytics import YOLO 
from retinaface import RetinaFace
from facenet_pytorch import MTCNN
from ultralytics.utils.plotting import Annotator, colors



class HumanFaceDetector:

    def __init__(self,source_video_directory, processed_video_directory, models_directory,allowed_video_extensions):
        self.face_detection_models = ["mtcnn","retinaFace","yolov8","yolov11","opencv"]
        self.source_video_directory = source_video_directory
        self.processed_video_directory = processed_video_directory
        self.models_directory = models_directory
        self.video_extensions = allowed_video_extensions

    def detect_humans_in_video(self, video_path, output_dir, csv_file):
        """
        Detect humans in video and save coordinates to CSV
        """
        # Initialize

        try:
            yolo_model_path = f"{self.models_directory}/yolo11n.pt"
            model = YOLO(yolo_model_path)
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), "Error reading video file"
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize DataFrame
            df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "image_path", "timestamp_in_video", 
                    "is_person", "has_face", "mtcnn_pt", "mtcnn_res", "retinaFace_pt", "retinaFace_res", 
                    "yolov8_pt", "yolov8_res", "yolov11_pt", "yolov11_res", "opencv_pt", "opencv_res"])
            
            df.index.name = 'id'  # Set the index name to 'id'
            df.index = df.index + 1  # Start the index from 1 instead of 0
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                # Get timestamp
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = str(timedelta(milliseconds=timestamp_ms))
                
                # Detect humans
                results = model.predict(frame, show=False)
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                
                if boxes:
                    human_detected = False  # Flag to track if a human is found in the frame
                    frame_path = None

                    # Mark detected human coordinates with green rectangle
                    marked_frame = frame.copy()
                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] == "person":
                            human_detected = True
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            label = f"Human {len(df)}"
                            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                            cv2.putText(marked_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if human_detected:
                        # Save marked frame
                        marked_human_path = f"{output_dir}/person_marked_frames/frame_{frame_count}.jpg"
                        cv2.imwrite(marked_human_path, marked_frame)


                        # Save frame as image
                        frame_path = f"{output_dir}/person_frames/frame_{frame_count}.jpg"
                        cv2.imwrite(frame_path, frame)
                    
                    if frame_path:
                        # Store detections`
                        for box, cls in zip(boxes, clss):
                            if model.names[int(cls)] == "person":
                                new_row = {
                                    "frame": frame_count,
                                    "x1": int(box[0]),
                                    "y1": int(box[1]),
                                    "x2": int(box[2]),
                                    "y2": int(box[3]),
                                    "image_path": frame_path,
                                    "timestamp_in_video": timestamp
                                }
                                df.loc[len(df)] = new_row
                
                frame_count += 1
                
            cap.release()
            df.to_csv(csv_file, index=True)
            return csv_file
        except Exception as e:
            print(f"Error detecting humans in a video: {e}")
    
    def mtcnn_process_frame_faces(self, image, group, output_dir, frame_number):
        """Process faces in a single frame using MTCNN"""
        start_time = time.time()
        mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        faces_found = False
        
        try:
            updates = []
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]
                
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    person_region_rgb = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
                    face_boxes, _ = mtcnn.detect(person_region_rgb)
                    
                    if face_boxes is not None and len(face_boxes) > 0:
                        faces_found = True
                        face_found_in_person = True
                        for fbox in face_boxes:
                            x1, y1, x2, y2 = map(int, fbox)
                            if x1 < x2 and y1 < y2:
                                cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({"mtcnn_pt":round(time.time()-processing_time,4),"mtcnn_res":int(face_found_in_person),"id":row.id})
            
            output_path = None
            if faces_found:
                output_path = f"{output_dir}/marked_frame_{frame_number}.jpg"
                cv2.imwrite(output_path, image)
                
            return time.time() - start_time, output_path, updates
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return time.time() - start_time, None, updates

    def retinaFace_process_frame_faces_v1(self, image, group, output_dir, frame_number):
        """Process faces in person regions using RetinaFace, this first generate image segement of the person"""
        start_time = time.perf_counter()
        faces_found = False
        temp_dir = os.path.abspath(output_dir)
        temp_path = os.path.join(temp_dir, f"temp_{frame_number}.jpg")
        updates = []
        
        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]
                
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    os.makedirs(temp_dir, exist_ok=True)
                    cv2.imwrite(temp_path, person_region)
                    
                    if os.path.exists(temp_path):
                        faces = RetinaFace.detect_faces(temp_path)
                        
                        if isinstance(faces, dict) and len(faces) > 0:
                            faces_found = True
                            for face_id, face_data in faces.items():
                                facial_area = face_data["facial_area"]
                                x1, y1, x2, y2 = facial_area
                                cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({"retinaFace_pt":round(time.time()-processing_time,4),"retinaFace_res":int(face_found_in_person),"id":row.id})
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            output_path = None
            if faces_found:
                output_path = os.path.join(temp_dir, f"marked_frame_{frame_number}.jpg")
                cv2.imwrite(output_path, image)
                
            return time.perf_counter() - start_time, output_path, updates
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return time.perf_counter() - start_time, None, updates

    def retinaFace_process_frame_faces(self, image, group, output_dir, frame_number):
        """Process faces in person regions using RetinaFace"""
        start_time = time.perf_counter()
        faces_found = False
        updates = []
        try:
            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]
                
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    # Use RetinaFace.detect_faces directly
                    faces = RetinaFace.detect_faces(person_region)
                    print("Detect person region")                
                    if isinstance(faces, dict) and len(faces) > 0:
                        faces_found = True
                        face_found_in_person = True
                        print("Found face")
                        for face_id, face_data in faces.items():
                            facial_area = face_data["facial_area"]
                            x1, y1, x2, y2 = facial_area
                            cv2.rectangle(person_region, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({"retinaFace_pt":round(time.time()-processing_time,4),"retinaFace_res":int(face_found_in_person),"id":row.id})
            
            output_path = None
            if faces_found:
                output_path = os.path.join(output_dir, f"marked_frame_{frame_number}.jpg")
                cv2.imwrite(output_path, image)
                print(output_path)
                
            return time.perf_counter() - start_time, output_path, updates
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return time.perf_counter() - start_time, None, updates

    def yolo_process_frame_faces(self, image, group, output_dir, frame_number,model,confidence_threshold=0.25, imgsz=1280):
        """
        Process faces in a single frame using YOLOv11 for face detection within human bounding boxes.

        Args:
            image (numpy.ndarray): The input image.
            group (pd.DataFrame): DataFrame containing human bounding boxes for the current frame.
            output_dir (str): Directory to save the processed image.
            frame_number (int): Frame number for naming the output file.
            model_path (str): Path to the YOLOv11 face detection model.
            confidence_threshold (float): Confidence threshold for face detection (default: 0.25).
            imgsz (int): Image size for inference (default: 1280).

        Returns:
            tuple: Execution time and path to the saved image (if faces were detected).
        """
        start_time = time.time()
        faces_found = False

        model_path = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/yolo/yolov8n-face.pt"
        if model == "yolov11":
            model_path = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/yolo11/yolov11n-face.pt"

        try:
            # Load the YOLOv11 face detection model
            face_model = YOLO(model_path)
            updates = []

            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                # Extract the person region from the image
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                # Ensure the person region is large enough for face detection
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    # Run YOLOv11 face detection on the person region
                    results = face_model(person_region, imgsz=imgsz, conf=confidence_threshold)

                    # Process detected faces
                    for result in results:
                        boxes = result.boxes  # Access bounding boxes
                        if len(boxes) > 0:
                            faces_found = True
                            face_found_in_person = True
                            for box in boxes:
                                # Extract face bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                confidence = box.conf.item()  # Confidence score
                                class_id = int(box.cls.item())  # Class ID (0 for face)

                                # Draw a green rectangle around the face
                                cv2.rectangle(person_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"Face {confidence:.2f}"
                                cv2.putText(person_region, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Replace the original person region with the processed one
                    image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region
                updates.append({f"{model}_pt":round(time.time()-processing_time,4),f"{model}_res":int(face_found_in_person),"id":row.id})

            # Save the processed image if faces were found
            output_path = None
            if faces_found:
                output_path = f"{output_dir}/marked_frame_{frame_number}.jpg"
                cv2.imwrite(output_path, image)
                print(output_path)

            return time.time() - start_time, output_path, updates

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return time.time() - start_time, None, updates

    def opencv_process_frame_faces(self, image, group, output_dir, frame_number):
        """
        Process faces in a single frame using OpenCV's Haar Cascade for face detection within human bounding boxes.

        Args:
            image (numpy.ndarray): The input image.
            group (pd.DataFrame): DataFrame containing human bounding boxes for the current frame.
            output_dir (str): Directory to save the processed image.
            frame_number (int): Frame number for naming the output file.

        Returns:
            tuple: Execution time and path to the saved image (if faces were detected).
        """
        start_time = time.time()
        faces_found = False

        try:
            # Load the Haar Cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            updates = []

            for _, row in group.iterrows():
                face_found_in_person = False
                processing_time = time.time()
                # Extract the person region from the image
                person_region = image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]

                # Ensure the person region is large enough for face detection
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    # Convert the person region to grayscale (required for Haar Cascade)
                    person_region_gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the person region
                    faces = face_cascade.detectMultiScale(
                        person_region_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # Process detected faces
                    if len(faces) > 0:
                        faces_found = True
                        face_found_in_person = True
                        for (fx, fy, fw, fh) in faces:
                            # Draw a green rectangle around the face
                            cv2.rectangle(person_region, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                        # Replace the original person region with the processed one
                        image[int(row.y1):int(row.y2), int(row.x1):int(row.x2)] = person_region

                updates.append({"opencv_pt":round(time.time()-processing_time,4),"opencv_res":int(face_found_in_person),"id":row.id})
            # Save the processed image if faces were found
            output_path = None
            if faces_found:
                output_path = f"{output_dir}/marked_frame_{frame_number}.jpg"
                cv2.imwrite(output_path, image)

            return time.time() - start_time, output_path, updates

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return time.time() - start_time, None, updates
    
    def detect_face_in_human_frames(self, csv_file, output_dir,model):
        """Process frames from CSV and track metrics"""
        fastest, slowest, total_time, total_images, faces_count = 0, 100, 0, 0, 0
        
        try:
            # Read and group DataFrame
            df = pd.read_csv(csv_file)
            grouped = df.groupby(['image_path', 'frame'])
            total_images = len(grouped)
            
            for (image_path, frame), group in grouped:
                if not os.path.exists(image_path):
                    continue
                image = cv2.imread(image_path)
                
                if(model == "mtcnn"):
                    execution_time, new_path, updates = self.mtcnn_process_frame_faces(image, group, output_dir, frame) #MTCNN: Multi-task Cascaded Convolutional Neural Network
                elif(model == "retinaFace"):
                    execution_time, new_path, updates = self.retinaFace_process_frame_faces(image, group, output_dir, frame) #Retina Face
                elif(model == "opencv"):
                    execution_time, new_path, updates = self.opencv_process_frame_faces(image, group, output_dir, frame) #OpenCV
                elif(model == "yolov8" or model == "yolov11"):
                    execution_time, new_path, updates = self.yolo_process_frame_faces(image, group, output_dir, frame, model) #Yolo v8 or v11 Face model

                self.update_dataframe(updates, csv_file)        
                
                if new_path:
                    faces_count += 1
                total_time += execution_time
                slowest = max(slowest, execution_time)
                fastest = min(fastest, execution_time)
            
            average_time = total_time/total_images if total_images > 0 else 0
            return fastest, slowest, total_time, total_images, average_time, faces_count
        except Exception as e:
            print(f"Error detecting face in human frames: {e}")
            return 0, 0, 0, 0, 0, 0

    def find_file_extensions(self, root_folder,extensions):
        """
        Recursively search for a particular file extension in the given folder and yield the containing folder,
        last two segments of the file name without the extension, and the absolute path.
        
        Args:
            root_folder (str): The root directory to start the search.
        
        Yields:
            tuple: Containing folder name, last two segments of the file name, and absolute path.
        """
        try:
            for dirpath, _, filenames in os.walk(root_folder):
                for file in filenames:
                    if file.lower().endswith(tuple(extensions)): 
                        # Extract the containing folder name
                        folder_name = os.path.basename(dirpath)
                        
                        # Remove the extension and split the file name
                        file_name_without_ext = os.path.splitext(file)[0]
                        # segments = file_name_without_ext.rsplit("_", 2)
                        # last_two_segments = "_".join(segments[-2:]) if len(segments) >= 2 else file_name_without_ext
                        
                        # Get the absolute path of the file
                        absolute_path = os.path.join(dirpath, file)
                        
                        # Yield the results
                        yield folder_name, file_name_without_ext, absolute_path

        except Exception as e:
            print(f"Error looping through directory: {e}")
    
    def create_folder_structure(self):
        try:
            for folder, segments, video_path in self.find_file_extensions(self.source_video_directory,self.video_extensions):
                # print(folder, segments, video_path)
                save_folder = f"{self.processed_video_directory}/{segments}"
                os.makedirs(save_folder, exist_ok=True) #create the folder for a particular video
                os.makedirs(save_folder+"/person_frames", exist_ok=True) #This is the folder that the face detectors will look at
                os.makedirs(save_folder+"/person_marked_frames", exist_ok=True) #This is the same data but the person found will be marked so that humans can see it
                Path(save_folder+"/info.txt").touch(exist_ok=True) #file that keep information about the video processing
                Path(save_folder+"/detected_humans.csv").touch(exist_ok=True) #csv file for detected humans
                for model in self.face_detection_models:
                    os.makedirs(save_folder+f"/{model}", exist_ok=True)

        except Exception as e:
            print(f"Error creating file: {e}")

    def update_dataframe(self, updates, file_path):
        df = pd.read_csv(file_path)
        for update in updates:
            id = update["id"]
            for col, value in update.items():
                if col != "id":  # Skip the "id" key
                    df.loc[id, col] = value

        df.to_csv(file_path, index=False)
        
    def run_human_detection(self):
        try:
            for folder, segments, video_path in self.find_file_extensions(self.source_video_directory,self.video_extensions):
                save_folder = f"{self.processed_video_directory}/{segments}"
                csv_file = save_folder+"/detected_humans.csv"

                start_time = time.time()
                self.detect_humans_in_video(video_path, save_folder, csv_file)
                total_time = time.time()-start_time

                info_text = f"Processed time: {total_time}"
                self.write_to_file(save_folder+"/info.txt",info_text)

        except Exception as e:
            print(f"Error in run_human_detection: {e}")

    def run_face_detection(self):
        try:
            for folder, segments, video_path in self.find_file_extensions(self.source_video_directory,self.video_extensions):
                save_folder = f"{self.processed_video_directory}/{segments}"
                csv_file = save_folder+"/detected_humans.csv"
                for model in hfd.face_detection_models:
                    fastest, slowest, total_time, total_images, average_time, faces_count = self.detect_face_in_human_frames(csv_file, save_folder+f"/{model}", model)
                    info_text = f"Face Detection Details: \n{fastest=}, \n{slowest=}, \n{total_time=}, \n{total_images=}, \n{average_time=},\n{faces_count=}"
                    self.write_to_file(save_folder+"/info.txt",info_text)

        except Exception as e:
            print(f"Error in run_face_detection: {e}")

    def write_to_file(self, file_path, text):
        """
        Writes the given text to the specified file on a new line.

        Args:
            file_path: The path to the file.
            text: The text to be written to the file.
        """
        try:
            with open(file_path, 'a') as file:  # Open the file in append mode ('a')
                file.write(text + '\n')  # Write the text followed by a newline character
                print(f"Successfully wrote '{text}' to '{file_path}'")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            # Create the file here if it doesn't exist
            with open(file_path, 'w') as file:
              file.write(text + '\n')





if __name__ == "__main__":

    source_video_directory = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/zf_source_videos"
    processed_video_directory = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/zf_processed_videos"
    # source_video_directory = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/new_source"
    # processed_video_directory = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/new_processed"
    
    models_directory = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/models"
    # allowed_video_extensions = [".mkv",".mp4"]
    allowed_video_extensions = [".mkv"]

    hfd = HumanFaceDetector(source_video_directory, processed_video_directory, models_directory, allowed_video_extensions)

    hfd.create_folder_structure()
    hfd.run_human_detection()
    hfd.run_face_detection()
