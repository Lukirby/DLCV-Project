# DLCV-Project

Synthetic images obtained from https://download.visinf.tu-darmstadt.de/data/from_games/
Real images for adaptation obtained from https://www.cityscapes-dataset.com/

Here it is described the folder structure of the project:

```
DLCV-Project/
├── cityscapes/            # Cityscapes dataset
│   ├── gtFine/            # Ground truth fine annotations
│   │   ├── test           # Ground truth test not available
│   │   ├── train
│   │   ├── val
│   └── leftImg8bit/
│       ├── test
│       ├── train
│       └── val
├── DeepLabV3/             # Train and inference scripts of DeepLabV3 model
├── DomainAdaptation/      # Adaptation scripts of both models DeepLabV3 and U-Net
├── models/                # Saved checkpoints and model weights
├── syn_resized_gt/        # Synthetic resized ground truth images from GTA-V dataset
├── syn_resized_image/     # Synthetic resized images from GTA-V dataset
├── Unet/                  # Train and inference scripts of U-Net model
├── utils/                 # Utility functions and scripts
├── visualization/         # Visualization scripts for qualitative analysis
└── README.md              # This README file
```

- **DeepLabV3/**: Contains the implementation of the DeepLabV3 model, including training and inference scripts:
  - `DeepLabV3_*.ipynb`: The script where a DeepLabV3 model with * variation is trained and validated.

- **DomainAdaptation/**: Contains the implementation of domain adaptation techniques for both models:
  - `DomainAdaptation_*.ipynb`: The script where domain adaptation techniques are applied to the model *.

- **Unet/**: Contains the implementation of the U-Net model, including training and inference scripts:
  - `Unet_*.ipynb`: The script where a U-Net model with * variation is trained and validated.

- **utils/**: Contains utility functions and scripts that support the main implementations:
  - `resize.ipynb`: Contains functions for resizing images and annotations, used for storage optimization.
  - `labelIdExtractor.ipynb`: The script where label IDs from ground truth annotations were converted.

- **visualization/**: Contains scripts for visualizing the results of the models:
  - `Visualization_DA_*.ipynb`: The script where the results of the domain adaptation techniques on * model are printed.
  - `Visualization_simulation.ipynb`: The script where the segmentation results of the models are visualized.