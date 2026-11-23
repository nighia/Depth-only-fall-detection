
# System architecture

```mermaid
flowchart TD
 A[RGB camera/video] --> B[Frame reader]
 B --> C[YOLO26 Depth]
 C --> D[Numeric depth map]
 D --> E[Depth background model]
 E --> F[Connected components]
 F --> G[Region filtering]
 G --> H[Person-region tracker]
 H --> I[Depth feature extraction]
 I --> J[Temporal window builder]
 J --> K[GRU classifier]
 K --> L[Stage A]
 L --> M{Fall?}
 M -- Yes --> N[Stage B subtype]
 M -- No --> O[No fall event]
 N --> P[Event manager]
 O --> P
 P --> Q[FastAPI/dashboard]
```

Depth estimation outputs a dense map, not person masks or IDs. The initial region extractor uses fixed-camera depth background subtraction; it may fail when people merge, stand still, or overlap furniture/stairs. Evaluate a segmentation alternative separately. If it uses RGB semantic features, document that the downstream representation is no longer strictly depth-only.

Offline research and online inference must share the same preprocessing and feature schema.
