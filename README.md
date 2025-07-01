<!-- TABLE OF CONTENTS -->

## 🚩 Table of Contents

<details>
  <ol>
	<li>
	  <a href="#about-the-project">About The Project</a>
	</li>
	<li>
	  <a href="#getting-started">Getting Started</a>
	  <ul>
		<li><a href="#prerequisites">Prerequisites</a></li>
		<li><a href="#installation">Installation</a></li>
	  </ul>
	</li>
	<li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## 🧠 About The Project

This project is an advanced Optical Character Recognition (OCR) tool that combines a **ResNet** backbone and a **Transformer-based** sequence decoder. It is designed to improve OCR accuracy for complex and variable-length text by leveraging powerful sequence modeling capabilities of transformers.

🔍 Key Features:

-   ✅ Uses **pre-trained ResNet** as a visual feature extractor (CNN backbone).
-   ✅ Applies a **Transformer decoder** for modeling output text sequences.
-   ✅ Enhances prediction quality with the **Beam Search** algorithm.
-   ✅ Built on top of the popular **[VietOCR](https://github.com/pbcquoc/vietocr)** repository.

This tool is well-suited for OCR tasks involving multilingual or structured text, especially in Vietnamese.

---

<p align="right"><a href="#readme-top">⬆️</a></p>

<!-- GETTING STARTED -->

## 🚀 Getting Started

To get a local copy up and running, follow these steps.

### 📋 Prerequisites

Before you begin, ensure that you have the following installed:

-   **Python >= 3.8**
-   **pip**
-   **git**
-   **CUDA (if using GPU)**

You can verify your Python version:

```bash
python --version
```

### 📦 Installation

1. Clone the repository [tfmOCR](https://github.com/Kant2510/tfmOCR)
    ```bash
    git clone https://github.com/Kant2510/tfmOCR.git
    cd tfmOCR
    ```
2. (Optional) Create a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

---

<p align="right"><a href="#readme-top">⬆️</a></p>

<!-- USAGE EXAMPLES -->

## 📜 Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

### 🚀 Train

```python
from tfmOCR.tool.config import Cfg
from tfmOCR.model.trainer import Trainer

import torch
dataset_params = {
    'name':'data'
}
config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
         'print_every':200,
         'valid_every':400,
          'iters':3600
         }
config['dataset'].update(dataset_params)
config['dataset']['data_root'] = '/path/to/dataset'
config['trainer'].update(params)
config['pretrain'] = '/path/to/pretrain.pth'
config['weights'] = 'path/to/weights.pth'
config['predictor']['beamsearch'] = False

trainer = Trainer(config, pretrained=True)
trainer.config.save('defaults.yml')
trainer.train()
```

### 🤖 Predict

```python
config = Config.load_config('defaults.yml')

config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', config['device'])
config['weights'] = "/path/to/weights.pth"
config['dataset']['data_root'] = "/path/to/dataset"
config['dataset']['annotation'] = '/path/to/annotation'

config['predictor']['beamsearch'] = True
predictor = Predictor(config)
img = "/path/to/img"
img = Image.open(img)
plt.imshow(img)
s = predictor.predict(img, return_prob=False)
print(s)
plt.show()
```

<p align="right"><a href="#readme-top">⬆️</a></p>
