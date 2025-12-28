# Maritime Near-Miss Detection System

## A Physically-Aware Deep Learning Framework | ML & DL Course Project

## Introduction

This project, developed for my **Machine Learning and Deep Learning** course, addresses the challenge of maritime safety by transitioning from reactive risk classification to proactive trajectory prediction. Focusing on high-traffic areas like **Piraeus Port**, I developed a framework that leverages deep learning to forecast vessel movements with high precision. Unlike traditional "Black Box" models, my system utilizes a **"Glass Box" architecture**, integrating Newtonian kinematics (CPA/TCPA) to ensure that every safety alert is physically justifiable and auditable.

## Key Results

| Metric | Value |
|:---|:---|
| Mean Position Error | **7.98 meters** |
| Improvement | **500x** (from ~4 km baseline) |
| Prediction Horizon | 1 minute |
| Near-Miss Detection Rate | 15% of encounters |

## Project Logic

I built the system on a modular 3-step pipeline to ensure logical clarity:

1. **AI Module**: A 2-Layer LSTM (128 units) engine predicts future vessel coordinates by learning delta-displacement patterns.
2. **Physics Module**: A deterministic layer that calculates the Closest Point of Approach (CPA) and Time to CPA (TCPA) based on my AI predictions.
3. **Logic Module**: A rule-based classifier that triggers "Near-Miss" alerts based on maritime safety standards (CPA < 0.5 nm).

## Installation

### For Users

1. Ensure you have a browser with Jupyter Notebook capability (e.g., Anaconda or Google Colab).
2. Navigate to the `notebooks/` folder and open `Demo.ipynb`.
3. Run the cells to view my interactive trajectory plots and automated risk detection reports.

### For Contributors

1. Clone the repository:
```bash
   git clone https://github.com/pollyxinjei-byte/Near-Miss_Project.git
   cd Near-Miss_Project
```

2. Install the required technical stack:
```bash
   pip install -r requirements.txt
```

## Project Structure
```text
Near-Miss_Project/
├── src/
│   ├── __init__.py                # Package initializer (REQUIRED)
│   ├── model.py                   # LSTM architecture (REQUIRED)
│   ├── piraeus_loader.py          # AIS data preprocessing
│   ├── cpa_tcpa_vectorized.py     # Physics engine for CPA/TCPA
│   ├── decision_logic.py          # Binary risk classification (REQUIRED)
│   ├── train.py                   # Main training pipeline
│   ├── best_lstm_model.pth        # Trained weights (128 units)
│   ├── trajectory_prediction.png
│   └── error_distribution.png
├── notebooks/
│   └── Demo.ipynb                 # Interactive demonstration
├── presentation.pdf
├── report.pdf
├── requirements.txt
└── README.md
```

## Known Issues & Roadmap

- **Environmental Factors**: My current model focuses on human intent and inertia; it does not yet account for wind or current forces.
- **GNN Integration**: I am researching Graph Neural Networks to model complex multi-vessel interactions for future iterations.
- **Tactical Horizon**: I optimized the system for 1-minute windows; longer intervals are currently being validated.

## Course Information

- **Course**: Machine Learning and Deep Learning
- **Project Title**: From Classification to Prediction: A Physically-Aware Deep Learning Framework for Maritime Near-Miss Detection
- **Developer**: Polly Chen-Goujard
- **Program**: Master Program: AI for Business, 2025

## Support

If you find my project useful for your research or coursework, please give this repository a ⭐️!
