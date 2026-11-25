## Dataset Used in WasteVision

For this project, I used the **TrashNet dataset** to train and evaluate the waste classification model.

The dataset contains images belonging to six waste categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

### Dataset Preparation

The original dataset contains 2,527 images. I organized the images into their respective category folders and used a preprocessing script to prepare them for model training.

During preprocessing, six conflicting duplicate files were excluded from the dataset split without deleting the original images.

### Dataset Split

The processed dataset contains 2,521 images, divided as follows:

| Dataset | Number of Images |
|---|---:|
| Training | 1,764 |
| Validation | 378 |
| Testing | 379 |
| **Total** | **2,521** |

The dataset preparation process is handled by `prepare_data.py`. The generated split information is saved in `results/splits.json`.

### Dataset Source and Attribution

Dataset: [TrashNet on Hugging Face](https://huggingface.co/datasets/garythung/trashnet/tree/main)

Original creators: **Gary Thung and Mindy Yang**.

The original dataset license notice is available in the [TrashNet MIT License](https://github.com/garythung/trashnet/blob/master/LICENSE).

The dataset images are stored locally and are not included in this GitHub repository.