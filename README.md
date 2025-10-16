# AIRA & NoRA: Advanced Parameter-Efficient Fine-Tuning Methods

This repository contains the official implementation of two novel parameter-efficient fine-tuning methods for large models, both accepted at **ICCV 2025** (CCF-A, Top Conference in Computer Vision).

## 📚 Papers

### AIRA: Activation-Informed Low-Rank Adaptation for Large Models

**Authors:** Lujun Li, Dezhi Li, Cheng Lin, Wei Li, Wei Xue, Sirui Han, Yike Guo

**Conference:** International Conference on Computer Vision (ICCV) 2025

**Paper:** [ICCV 2025 Proceedings](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Efficient_Fine-Tuning_of_Large_Models_via_Nested_Low-Rank_Adaptation_ICCV_2025_paper.pdf)

AIRA introduces three key innovations for efficient fine-tuning:

- **Outlier-weighted SVD initialization** - Leverages activation patterns for better initialization
- **Outlier-driven dynamic rank assignment** - Adaptively allocates rank based on layer importance
- **Activation-informed training** - Incorporates activation statistics to guide the training process

![AIRA_Poster](AIRA_Poster.png)

### NoRA: Efficient Fine-Tuning of Large Models via Nested Low-Rank Adaptation

**Authors:** Lujun Li, Cheng Lin, Dezhi Li, You-Liang Huang, Wei Li, Tianyu Wu, Jie Zou, Wei Xue, Sirui Han, Yike Guo

**Conference:** International Conference on Computer Vision (ICCV) 2025

**Paper:** [ICCV 2025 Proceedings](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Efficient_Fine-Tuning_of_Large_Models_via_Nested_Low-Rank_Adaptation_ICCV_2025_paper.pdf)

NoRA presents a novel nested parameter-efficient LoRA structure that optimizes large model fine-tuning by employing serial structures for improved efficiency and effectiveness.

![NoRA_poster](NoRA_poster.png)

## 🚀 Getting Started

### Prerequisites

Before running the code, ensure you have the necessary dependencies installed. Please refer to the requirements in the repository for specific package versions.

### Dataset Setup

Download and prepare the required datasets before running experiments. For detailed instructions on downloading and preparing the datasets, please refer to DATASET.md.

## 💻 Usage

### Running AIRA & NoRA

To run the AIRA & NoRAcode with your dataset, use the following command:

bash

```bash
python main_XXX_A.py --root_path /path/to/your/data --dataset dtd --seed 1
```

**Parameters:**

- `--root_path`: Path to your data directory
- `--dataset`: Dataset name (e.g., dtd)
- `--seed`: Random seed for reproducibility

### Additional Configuration

For more configuration options and advanced usage, please refer to the command-line arguments in `main_XXX.py`.

## 📊 Key Features

### AIRA Features

- Efficient parameter-efficient fine-tuning using activation-informed strategies
- Dynamic rank allocation based on layer importance
- Improved initialization through outlier-weighted SVD

### NoRA Features

- Nested low-rank adaptation structure for enhanced efficiency
- Serial architecture design for optimized parameter usage
- Effective fine-tuning with reduced computational overhead

## 📖 Citation

If you find our work useful in your research, please consider citing:

### AIRA

```text
@InProceedings{Li_2025_ICCV,
    author    = {Li, Lujun and Li, Dezhi and Lin, Cheng and Li, Wei and Xue, Wei and Han, Sirui and Guo, Yike},
    title     = {AIRA: Activation-Informed Low-Rank Adaptation for Large Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1729-1739}
}
```

### NoRA

```text
@InProceedings{Li_2025_ICCV,
    author    = {Li, Lujun and Lin, Cheng and Li, Dezhi and Huang, You-Liang and Li, Wei and Wu, Tianyu and Zou, Jie and Xue, Wei and Han, Sirui and Guo, Yike},
    title     = {Efficient Fine-Tuning of Large Models via Nested Low-Rank Adaptation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {22252-22262}
}
```

## 📄 License

Please refer to the LICENSE file in this repository for usage terms and conditions.

## 🤝 Contributing

We welcome contributions to improve this repository. Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or discussions about the methods, please open an issue in this repository or contact the authors through the paper's correspondence information.

## 🙏 Acknowledgments

We thank the reviewers and the computer vision community for their valuable feedback and support.