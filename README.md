# topical-decoding

## About The Project

This repository contains the implementation of a custom `LogitsProcessor` by Joschka Braun and Bálint Mucsányi, conducted as a research project at the University of Tübingen. 
The `LogitsProcessor` is designed to enhance the topical focus in abstractive summarization by adjusting logits of topic-relevant tokens.
Our approach introduces three methods to promote topic-relevant tokens: 
- **Constant Shift**: Adds a constant value to logits.
- **Factor Scaling**: Multiplies logits by a constant factor.
- **Threshold Selection**: Set logits above a specific probability threshold to the maximum logit value in the current logit distribution.

The evaluation of these methods was carried out on the NEWTS dataset against the baseline strategy of prompt engineering with topic-associated keywords.

### Abstractive Topical Summarization Using Logits Reweighting

Abstractive topical summarization with autoregressive transformer-based language models poses significant challenges, especially when generating contextually relevant summaries without model retraining.
Our research proposes a more easily implementable, universally applicable, and resource-efficient method that enhances the topical focus of summaries without the need for extensive training or fine-tuning.

**Authors:** Joschka Braun & Bálint Mucsányi

**Supervisor:** Seyed Ali Bahrainian

## Getting Started

### Prerequisites

Before running the project, ensure you have the necessary Python environment and dependencies set up. See `pyproject.toml` for the list of required libraries.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/JoschkaCBraun/topical-decoding.git
```

Navigate to the project directory:
```bash
cd topical-decoding
```

Install the necessary Python packages using Poetry:
```bash
poetry install
```

### LDA_250 Model
The LDA_250 model is essential for running this project but is not included in the repository due to its size.
It is hosted on Google Drive and must be downloaded and placed in the data/ directory.

Download the LDA_250.zip file from the following Google Drive link:

https://drive.google.com/drive/folders/1Xf8j46y1FJ8vbZX1kqbDGhIiXoRON7VU?usp=sharing

After downloading, unzip the file and place the resulting LDA_250 folder in the topical-decoding/data/ directory of your local project setup.

Your data/ directory structure should now look like this:

```kotlin
data/
└── LDA_250/
    ├── dictionary.dic
    ├── lda.model
    ├── lda.model.expElogbeta.npy
    ├── lda.model.state
    └── lda.model.state.sstats.npy
```

## Usage
You can set the hyperparameters of the experiment you want to run in the config.py file. 
Run the baseline experiment by running the src/baseline/baseline.py file
Run the constant shift, factor scaling or threshold selection experiment by selecting it in the config.py file and running the src/logits_reweighting/logits_reweighting.py file.
Plot the results in the experiment in the notebooks/plot_results.ipynb notebook.

```python
# Run baseline
python src/baseline/baseline.py

# Run logits reweighting
python src/logits_reweighting/logits_reweighting.py

# Plot results
jupyter notebook notebooks/plot_results.ipynb
```

## License
This project is provided for educational purposes and is not licensed for commercial use.

For more details on the project's methodology, findings, and implications, please refer to the accompanying project report: logits_reweighting_project_report.pdf or contact the authors.