# Manufacturing Defect Analyzer

AI-powered visual inspection app that classifies surface defects on
hot-rolled steel strips. Upload an image, get the predicted defect type,
a confidence score, an explanation and a downloadable inspection report.

Built as a demonstration of computer vision for industrial quality control.

## Defect classes

Six classes from the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
(1,800 grayscale images, 300 per class):

| Class | Description |
|---|---|
| Crazing | Network of fine thermal-stress cracks |
| Inclusion | Non-metallic particles embedded in the surface |
| Patches | Irregular regions with different texture/color |
| Pitted Surface | Small corrosion/gas cavities |
| Rolled-in Scale | Oxide scale pressed in during hot rolling |
| Scratches | Linear mechanical surface damage |

## Screenshots

Analysis result with prediction, confidence and recommendation:

![Analysis result](docs/screenshot.png)

Generated HTML inspection report:

![Inspection report](docs/screenshot2.png)

## Architecture

- **Model:** ResNet18 (ImageNet pretrained), final layer replaced with 6 classes
- **Training:** transfer learning, Adam, cross-entropy, ~15 epochs
- **App:** Streamlit — upload, predict, explain, HTML report

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data preparation

1. Download the NEU-DET dataset from Kaggle
   (`kaggle datasets download kaustubhdikshit/neu-surface-defect-database`)
   and extract it into `data/raw/`.
2. Split into train/val/test (70/15/15):

```bash
python model/prepare_data.py
```

## Training

```bash
python model/train.py --epochs 15
```

The best checkpoint (by validation accuracy) is saved to
`model/saved_models/resnet18_neu.pt`, plus a `.metrics.json` with the
test-set classification report.

## Running the app

```bash
streamlit run app/streamlit_app.py
```

## CLI prediction

```bash
python model/predict.py data/sample_images/scratches_1.jpg
```

## Project structure

```
├── app/                  # Streamlit UI, report generator, defect info
├── model/                # training, prediction, data prep
│   └── saved_models/     # checkpoints (gitignored)
├── data/
│   ├── raw/              # extracted NEU-DET dataset (gitignored)
│   ├── processed/        # train/val/test split (gitignored)
│   └── sample_images/    # a few demo images for the app
├── notebooks/            # exploratory analysis
└── reports/              # generated inspection reports
```

## Future improvements

- YOLO-based defect localization (NEU-DET ships with bounding-box
  annotations, so the same dataset covers detection)
- Real-time camera inspection
- Dashboard for defect trends across analyzed batches
- FastAPI backend + PostgreSQL storage
- PDF reports, role-based login, Docker deployment
