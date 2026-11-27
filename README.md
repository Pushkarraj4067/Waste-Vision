# WasteVision
### Deep Learning Based Waste Image Classification

WasteVision is a computer vision project that classifies waste images into six categories: cardboard, glass, metal, paper, plastic and trash.

The project uses Python, PyTorch and ResNet18 to explore image classification through transfer learning. It includes dataset preparation, model training, evaluation and a Streamlit application for testing images.

## Project Overview

Proper waste identification is an important part of waste management. WasteVision explores how deep learning can help identify common waste materials from images.

The system takes an image as input, processes it through a trained convolutional neural network and predicts its corresponding waste category.

## Features

- Classification of six waste categories.
- ResNet18-based image classification.
- Transfer learning using pretrained ImageNet weights.
- Dataset preparation and stratified splitting.
- Model training and validation.
- Accuracy, precision, recall and F1-score evaluation.
- Confusion matrix and prediction analysis.
- Single-image prediction.
- Interactive Streamlit interface.
- Saved model checkpoint for inference.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet18 and image transformations |
| NumPy | Numerical operations |
| Scikit-learn | Model evaluation |
| Matplotlib | Training visualizations |
| Streamlit | Web application |
| Pillow | Image processing |

## Dataset

The project uses the **TrashNet dataset**, collected by Gary Thung and Mindy Yang.

The dataset contains 2,527 images across six categories:

- Cardboard: 403 images
- Glass: 501 images
- Metal: 410 images
- Paper: 594 images
- Plastic: 482 images
- Trash: 137 images

Six files in three byte-identical pairs with conflicting labels were excluded.

The remaining 2,521 images were divided using stratified splitting:

| Dataset | Images |
|---|---:|
| Training | 1,764 |
| Validation | 378 |
| Testing | 379 |
| Total | 2,521 |

The dataset is not included in this GitHub repository. Refer to `data/README.md` for download instructions.

## Model Architecture

WasteVision uses ResNet18 with transfer learning.

The pretrained ImageNet backbone is frozen, and its final fully connected layer is replaced with a new classification layer containing six output classes.

**Training configuration:**

| Parameter | Value |
|---|---|
| Model | ResNet18 |
| Epochs | 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Input size | 224 × 224 |
| Training device | CPU |
| Output classes | 6 |

Only the final classification layer is trained.

## Results and Evaluation

The selected model achieved the following results on the held-out test dataset.

<details>
<summary><strong>Model performance</strong></summary>

| Metric | Result |
|---|---:|
| Test Accuracy | 81.00% |
| Macro Precision | 79.38% |
| Macro Recall | 76.03% |
| Macro F1-score | 77.17% |

The model correctly classified 307 out of 379 test images.

</details>

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 98.00% | 80.33% | 88.29% |
| Glass | 71.26% | 82.67% | 76.54% |
| Metal | 81.36% | 77.42% | 79.34% |
| Paper | 79.25% | 94.38% | 86.15% |
| Plastic | 90.16% | 76.39% | 82.71% |
| Trash | 56.25% | 45.00% | 50.00% |

The trash category had the weakest performance, partly because it had fewer training and testing examples.

The results show that overall accuracy alone does not fully represent model performance. Class-wise precision, recall and F1-score are also important.

## Application

The project includes a Streamlit application that allows users to upload an image and receive a predicted waste category.

The application displays:
- Uploaded image preview.
- Predicted waste category.
- Model prediction score.
- Top three predictions.

The application uses the same prediction function as the evaluation pipeline.

## Project Structure

```text
WasteVision/
│
├── app.py
├── app_new.py
├── train.py
├── evaluate.py
├── utils.py
├── prepare_data.py
├── download_dataset.py
│
├── data/
│   └── README.md
│
├── models/
│   └── wastevision_stage3.pth
│
├── results/
│   ├── history.csv
│   ├── splits.json
│   └── stage4/
│
├── requirements.txt
└── README.md
```

Note: The structure above highlights the main project files. Check the repository for the complete directory listing.

## Installation

Clone the repository:

```bash
git clone https://github.com/Pushkarraj4067/Waste-Vision.git
```

Move into the project directory:

```bash
cd Waste-Vision
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install the dependencies:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

Upload a JPG, JPEG or PNG image and click **Classify Image** to view the prediction.

## Training and Evaluation

To reproduce the training workflow, prepare the TrashNet dataset as described in `data/README.md`.

Run the following commands from the project root:

```bash
python prepare_data.py
python train.py
python evaluate.py
```

The training script saves the checkpoint with the lowest validation loss. The evaluation script uses the saved checkpoint to calculate test metrics and generate evaluation outputs.

## Limitations

- The dataset is relatively small and imbalanced.
- The trash category has fewer examples than other categories.
- Most dataset images have plain backgrounds.
- Real-world lighting and background conditions may affect predictions.
- The model predicts one category per image.
- Multiple objects are not detected separately.
- Unrelated images may still receive one of the six available labels.
- Prediction scores are not calibrated probabilities.

## Future Improvements

Some possible improvements include:

- Increasing the size and diversity of the dataset.
- Improving performance on underrepresented categories.
- Adding unknown-image detection.
- Supporting multiple waste objects in one image.
- Exploring object detection models.
- Improving performance on real-world waste images.

## What I Learned

Through this project, I explored:

- Image classification using deep learning.
- Transfer learning with pretrained models.
- Dataset preparation and stratified splitting.
- Training and validation workflows.
- Model evaluation using multiple metrics.
- Confusion matrix analysis.
- Integrating a trained model with Streamlit.
- Understanding model limitations and generalization.

## Acknowledgements

The project uses the dataset collected by Gary Thung and Mindy Yang.

Dataset source: https://github.com/garythung/trashnet

The project also uses PyTorch, Torchvision and other third-party libraries. Their respective licences and pretrained-weight terms apply.

---

**Project:** WasteVision  
**Developer:** Pushkarraj Gattani  
**GitHub:** https://github.com/Pushkarraj4067/Waste-Vision
