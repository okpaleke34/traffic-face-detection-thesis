
# Traffic Face Detection Thesis Project

This repository contains the code and resources for my final thesis project: **Study of selected open-source algorithms for face detection in image sequence**. This project evaluates the performance of various open-source face detection algorithms in the context of Advanced Driver Assistance Systems (ADAS).

## Thesis Context

**Thesis Title:** Study of selected open-source algorithms for face detection in image sequence

**Author:** Chukwudi OKPALEKE

**Specialization:** Informatics

**Supervisor:** Dr. Hab. Eng. Henryk Palus, Prof. PS

**Department:** Control, Electronics and Information Engineering

**Faculty:** Faculty of Automatic Control, Electronics and Computer Science

**Abstract:**

This thesis investigates the performance of selected open-source algorithms for face detection in image sequences, specifically addressing their applicability in Advanced Driver Assistance Systems (ADAS). The study is motivated by the need to reliably detect faces in diverse traffic conditions, as image sequences from car-mounted cameras often capture the faces of pedestrians and other individuals. The research explores the practical challenges of deploying such algorithms in real-world ADAS applications.

**Methodology:** This research involves a systematic review of open-source face detection algorithms available online. The selected algorithms are evaluated based on their ability to detect faces under varying skin tones, clothing, and lighting conditions, aligning with the robustness requirements of ADAS.

## Getting Started

To run this project, you will need to set up your environment and install the necessary dependencies.

### Prerequisites

* **Python 3.8 or higher:** Ensure you have Python 3.8 or a later version installed on your system. You can download it from [python.org](https://www.google.com/url?sa=E&source=gmail&q=https://www.python.org).
* **Pip:**  Python package installer, usually included with Python installations.

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/okpaleke34/traffic-face-detection-thesis.git](https://github.com/okpaleke34/traffic-face-detection-thesis.git)
   cd traffic-face-detection-thesis
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the required Python libraries listed in the `requirements.txt` file, including:

   * `pandas`
   * `opencv-python`
   * `torch`
   * `ultralytics` (YOLOv8, YOLOv11)
   * `retina-face`
   * `facenet-pytorch`
   * `seaborn`
   * `matplotlib`

### Downloading Models

The project uses pre-trained models for human and face detection. You need to download these models and place them in the `models` directory within the repository.

* **YOLO Models:** The project uses YOLOv11n.pt for human detection, and yolov8n-face.pt, yolov11n-face.pt for face detection. You can download these pre-trained weights from the official YOLO repositories or from the links provided in the thesis documentation if available and place them in the `models` folder.
* **OpenCV Haar Cascade:** The OpenCV Haar Cascade classifier (`haarcascade_frontalface_default.xml`) is typically included with OpenCV installation, but if not found, ensure your OpenCV installation is complete or download it and place it in the `models` directory.

**Ensure the following model files are present in the `models` directory:**

* `models/yolo11n.pt` (YOLOv11 human detection model)
* `models/yolov8n-face.pt` (YOLOv8 face detection model)
* `models/yolov11n-face.pt` (YOLOv11 face detection model)
* `models/haarcascade_frontalface_default.xml` (OpenCV Haar Cascade - should be in `cv2.data.haarcascades` by default)

## Running the Program

The main script to run the program is `main.py`. You can execute it from the project root directory.

```bash
python main.py
```

**Configuration:**

Before running, you might need to adjust the configuration settings within the `if __name__ == "__main__":` block in `main.py`.  Specifically, update the paths to your:

* `source_video_directory`:  Path to the directory containing your source video files.
* `processed_video_directory`: Path to the directory where you want to save the processed output (frames, CSV files, etc.).
* `models_directory`:  Path to the `models` directory within the repository (should be correct if you followed the installation steps).
* `allowed_video_extensions`:  List of video file extensions to be processed (e.g., `[".mp4", ".mkv", ".avi"]`).

**Example Configuration in `main.py`:**

```python

if __name__ == "__main__":
    config = {
        "source_video_directory": "/path/to/source/videos",
        "processed_video_directory": "/path/to/processed",
        "models_directory": "/path/to/model/files",
        "allowed_video_extensions": [".mp4", ".mkv", ".avi"]
    }

    hfd = HumanFaceDetectorOpt(**config)

    hfd.create_folder_structure()
    hfd.run_parallel_processing() # Run parallel processing
```

### Program Output

The program will process each video file in the `source_video_directory`, perform human and face detection using the configured models, and save the following outputs in the `processed_video_directory` under a folder named after each video segment:

* **`person_frames/`**: Directory containing frames where humans were detected.
* **`person_marked_frames/`**: Directory containing frames with bounding boxes drawn around detected humans.
* **`[face_detection_model_name]/`**: Directories (e.g., `mtcnn`, `retinaFace`, `yolov8`, `yolov11`, `opencv`) containing marked frames where faces were detected by each respective model.
* **`detected_humans.csv`**: CSV file containing detailed information about each detected human instance, including bounding box coordinates, timestamps, human/face presence annotations, and face detection model results and processing times.
* **`info.txt`**: Text file summarizing the processing time and performance metrics for human and face detection.

## Usage

1. **Prepare your video dataset:** Place your video files in the `source_video_directory`.
2. **Configure `main.py`:** Update the directory paths in the `if __name__ == "__main__":` block to match your setup.
3. **Run `main.py`:** Execute the script using `python main.py`.
4. **Review Results:**  Examine the output in the `processed_video_directory`. Analyze the `detected_humans.csv` file and the generated image frames to evaluate the performance of the face detection models.

## Repository Link

[https://github.com/okpaleke34/traffic-face-detection-thesis.git](https://github.com/okpaleke34/traffic-face-detection-thesis.git)

## License

This project is open-source and available under the [MIT License](https://www.google.com/url?sa=E&source=gmail&q=LICENSE) (if you have one, otherwise specify the license or remove this section).

---

**Contact:** Chukwudi OKPALEKE ( [okpaleke34.pl@gmail.com] )
