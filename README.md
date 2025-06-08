# HERMES: Healthcare Ethics & Robustness in Medical Image Systems

## Description

**HERMES** is a project focused on ensuring healthcare ethics and robustness in medical image systems. It is part of the Ethics in AI course at the [University of Bologna](https://www.unibo.it/en/homepage), and it aims to provide a framework for training and evaluating medical image models with a focus on ethical considerations and robustness.

Our main goal is to create a system that can be used easily to train and evaluate medical image models, while also providing a framework for ethical considerations and robustness. The project is designed to be **modular** and **extensible**, allowing for easy experimentation with different settings, training strategies, and data preprocessing techniques.

Machine Learning Security is crucial in the healthcare domain, thus we opted for a modular approach based on a simple **PyTorch Classifier** built on top of the [**PyTorch Lightning**](https://lightning.ai/docs/pytorch/stable/) framework.

As for the robustness of the model, we are using the [**Adversarial Robustness Toolbox (ART)**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) by the [**Linux Foundation AI & Data Foundation**](https://lfaidata.foundation/), which provides a comprehensive suite of tools for **adversarial machine learning**.

All the code is compatible with the API exposed by ART, allowing for easy integration with other tools and libraries and enabling the use of various adversarial attacks and defenses on the model.

The main focus of this project is based on **evasion attacks**, specifically the [**Fast Gradient Sign Method (FGSM)**](https://arxiv.org/abs/1412.6572) and the [**Projected Gradient Descent (PGD)**](https://arxiv.org/abs/1706.06083) attacks, which are widely used in the field of adversarial machine learning. The defenses implemented in this project are mainly **preprocessing** techniques, such as [**JPEG compression**](https://arxiv.org/abs/1705.02900), **Gaussian Smoothing**, and [**Spatial Smoothing**](https://arxiv.org/abs/1704.01155), which are designed to reduce the impact of adversarial attacks on the model.

## Notebooks

The project includes several **notebooks** that you can use to finetune a PyTorch Classifier on a medical image dataset, evaluate the model's performance, and test the robustness of the model against adversarial attacks. The notebooks are designed to be modular and extensible, allowing you to easily experiment with different settings, training strategies, and data preprocessing techniques.

The notebooks are located in the **`notebooks`**, and they include:

- **`resnet_finetuning.ipynb`**: A notebook for finetuning a ResNet model pretrained on ImageNet on a medical image dataset (**PatchCamelyon - PCAM**). It includes data loading, preprocessing, training, and evaluation steps with PyTorch Lightning for easy model training and evaluation without boilerplate code.
- **`simple_framework.ipynb`**: A notebook for evaluating the robustness of the trained model against adversarial attacks. It includes the implementation of FGSM and PGD attacks from ART, as well as the evaluation of the model's performance on adversarial examples. The notebook also includes preprocessing techniques such as JPEG compression, Gaussian Smoothing, and Spatial Smoothing to reduce the impact of adversarial attacks on the model. The proposed framework allows for easy integration with other defenses from [**`art.defences.preprocessor`**](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html) but you can find also the code to implement your **own** preprocessing techniques compliant with the ART API.

## Installation

> [!TIP]
> We strongly encourage using [astral-sh/uv](https://github.com/astral-sh/uv) to install the project requirements. This ensures that the correct versions of the dependencies are installed, and it can help avoid conflicts with other projects. Alternatively, you can use `pip` to install the requirements from the `requirements.txt` file.

To install the project, you can use the following command:

```bash
uv sync
```

This command will install all the required dependencies for the project, including PyTorch, PyTorch Lightning, and the Adversarial Robustness Toolbox (ART).
If you prefer to use `pip`, you can install the requirements from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Authors and acknowledgment

This project is developed by [Matteo Fasulo](https://github.com/MatteoFasulo), [Luca Babboni](https://github.com/ElektroDuck), and [Maxim Omelchenko](https://github.com/omemaxim) as part of the Ethics in AI course at the [University of Bologna](https://www.unibo.it/en/homepage).

A great shoutout to the community of the [**Linux Foundation AI & Data Foundation**](https://lfaidata.foundation/) for their work on the [**Adversarial Robustness Toolbox (ART)**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) and for providing a comprehensive suite of tools for adversarial machine learning.

There are many examples and tutorials available in the [**ART documentation**](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/index.html) but we believe that this project can help you to get started with the framework and to understand how to use it in the context of medical image systems.

## License

The project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Project status

The project is currently in active development. We welcome contributions and feedback from the community. If you have any suggestions or issues, please open an issue!
