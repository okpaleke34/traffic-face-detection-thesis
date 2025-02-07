from HumanFaceDectatorOpt import HumanFaceDetectorOpt


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