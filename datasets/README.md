# Dataset Directory

## YOLO Format Dataset Structure

Place your dataset in the following structure:

```
datasets/
└── custom/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img1.txt
        │   ├── img2.txt
        │   └── ...
        ├── val/
        │   ├── img1.txt
        │   └── ...
        └── test/
            └── ...
```

## Label Format

Each `.txt` file contains one object per line in the format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: integer class ID (0-indexed)
- `x_center`: normalized x coordinate of box center (0-1)
- `y_center`: normalized y coordinate of box center (0-1)
- `width`: normalized box width (0-1)
- `height`: normalized box height (0-1)

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

## Configuration

Update `configs/dataset.yaml` with your dataset path and class names.
