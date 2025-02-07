```mermaid
graph TD
    A[Program Start] --> B[Initialize HumanFaceDetector]
    B --> C[Create Folder Structure]
    C --> D[Discover Video Files]
    D --> E[Parallel Video Processing]
    
    E --> F[Process Each Video]
    F --> G[Human Detection Phase]
    G --> H[YOLOv11 Human Detection]
    H --> I[Save Frames & CSV]
    
    F --> J[Face Detection Phase]
    J --> K[Parallel Model Execution]
    K --> L1[MTCNN]
    K --> L2[RetinaFace]
    K --> L3[YOLOv8]
    K --> L4[YOLOv11]
    K --> L5[OpenCV]
    
    L1 --> M[Process Human Regions]
    L2 --> M
    L3 --> M
    L4 --> M
    L5 --> M
    
    M --> N[Thread Pool Execution]
    N --> O[Load Model]
    O --> P[Detect Faces]
    P --> Q[Draw Bounding Boxes]
    Q --> R[Update CSV]
    R --> S[Save Marked Frame]
    
    E --> T[Aggregate Results]
    T --> U[Generate Reports]
    U --> V[Program End]
    
    style A fill:#4CAF50,stroke:#388E3C
    style B fill:#2196F3,stroke:#1976D2
    style G fill:#FF9800,stroke:#F57C00
    style J fill:#9C27B0,stroke:#7B1FA2
    style L1 fill:#E91E63,stroke:#C2185B
    style L2 fill:#E91E63,stroke:#C2185B
    style L3 fill:#E91E63,stroke:#C2185B
    style L4 fill:#E91E63,stroke:#C2185B
    style L5 fill:#E91E63,stroke:#C2185B
    style T fill:#009688,stroke:#00796B