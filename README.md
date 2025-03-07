# Enhancing Vietnamese VQA through Curriculum Learning on Raw and Augmented Text Representations

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Conference](https://img.shields.io/badge/AAAI--25%20Workshop-DUI-green)](https://aaai.org/Conferences/AAAI-25/)

## Description

This is the **official repository** for our paper, *Enhancing Vietnamese VQA through Curriculum Learning on Raw and Augmented Text Representations*, accepted at the AAAI-25 Workshop on Document Understanding and Intelligence. This repository contains the codebase for improving **Vietnamese Visual Question Answering (VQA)** using **curriculum learning** on both raw and augmented textual data.

## Prerequisites

We assume you have **Anaconda** installed for managing the virtual environment. If not, you can download it from [here](https://www.anaconda.com/products/distribution).

To set up the environment:

1. **Create and activate a new Anaconda environment**:

   ```bash
   conda create -n vqa_env python=3.11
   conda activate vqa_env
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

We use two datasets for training: **ViVQA** and **OpenViVQA**. You can download them from:

- [ViVQA Dataset](https://huggingface.co/datasets/SEACrowd/vivqa)
- [OpenViVQA Dataset](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)

After downloading, organize your datasets as follows:

```
datasets/
│── OpenViVQA/
│── ViVQA/
```

Once the datasets are structured correctly, **update the `data_dir` parameter in `config.py` to match your local path**.

## Generating Paraphrases

Before training, paraphrased datasets need to be generated. Run the following command:

```bash
python3 generate_new_dataset.py \
    --train_filepath path/to/your/dataset.csv \
    --num_params 20 \
    --random_seed 59 \
    --save_filepath paraphrases.csv \
    --paraphrase_method mt5
```

### Explanation of Arguments:
- `--train_filepath`: Path to the input training dataset.
- `--num_params`: Number of paraphrases to generate per sample.
- `--random_seed`: Seed for reproducibility.
- `--save_filepath`: Output filename for storing paraphrases.
- `--paraphrase_method`: Paraphrase generation method (`mt5` or `gpt`).

## Training

Once the dataset is prepared, start training by running:

```bash
bash start_training.sh
```

### Explanation of Key Parameters:
- `--epochs`: Number of training epochs.
- `--patience`: Number of epochs without improvement before early stopping.
- `--n_text_paras`: Number of text paraphrases used for augmentation.
- `--text_para_thresh`: Threshold for selecting paraphrases.
- `--is_text_augment`: Enable/disable text augmentation.
- `--use_dynamic_thresh`: Enable/disable dynamic thresholding.
- `--start_threshold`: Initial value for dynamic thresholding.
- `--min_threshold`: Minimum value for dynamic thresholding.
- `--is_log_result`: Enable logging of training results.

For additional arguments, refer to `train.py` or run:

```bash
python3 train.py --help
```

## Citation

If you find our work or this repository useful, please cite our paper:

```bibtex
@misc{nguyen2025enhancingvietnamesevqacurriculum,
      title={Enhancing Vietnamese VQA through Curriculum Learning on Raw and Augmented Text Representations},
      author={Khoi Anh Nguyen and Linh Yen Vu and Thang Dinh Duong and Thuan Nguyen Duong and Huy Thanh Nguyen and Vinh Quang Dinh},
      year={2025},
      eprint={2503.03285},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.03285},
}
```