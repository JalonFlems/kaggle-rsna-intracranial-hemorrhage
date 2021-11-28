# **CSC490** | Dataset Report & ML Setup

### **Group** | ProofByExample

### **Team Members** | Davinder Jangra, Micah Flemming, Michelle Luo, Shubham Sharma

---

## **Dataset** | RSNA Intracranial Hemorrhage Challenge Dataset

- [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)
- [Intracranial Hemorrhage Detection - PNG Format 128](https://www.kaggle.com/guiferviz/rsna_stage1_png_128?select=stage_1_test_images)
- [Intracranial Hemorrhage Detection - PNG Formats (<=224)](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110840)

Micah has downloaded the dataset and shared it with the team through Google Drive.

## Paper Reference(s)

- [Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge](https://pubs.rsna.org/doi/10.1148/ryai.2020190211)
- [A Smart Machine Learning Model for the Detection of Brain Hemorrhage Diagnosis Based Internet of Things in Smart Cities](https://doi.org/10.1155/2020/3047869)
- [Advanced machine learning in action: identification of intracranial hemorrhage on computed tomography scans of the head with clinical workflow integration](https://doi.org/10.1038/s41746-017-0015-z)
- [Expert-level detection of acute intracranial hemorrhage on head computed tomography using deep learning](https://doi.org/10.1073/pnas.1908021116)

## Why this dataset?

Intracranial hemorrhages are a life threatening and common condition that has many serious health impacts (trauma, stroke, aneurysm, etc.). Since the diagnosis is often very difficult and needs immediate attention, being able to quickly identify the type of hemorrhage and where it is located can have life saving implications [[1]](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview).

We chose this dataset because we’re tackling an extremely important problem as intracranial brain hemorrhages result in ~10% of strokes in the US [[2]](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview). Also, this dataset allows us to work on multiple different tasks related to the dataset. The first task is to classify whether or not a brain hemorrhage exists in a given image. If a brain hemorrhage was classified in a given image, then the second task is to correctly classify the subtype for the images. The figure below is from the competition page on Kaggle which displays the 5 different types of subtypes to be classified [[3]](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview/hemorrhage-types).

<img src="subtypes-of-hemorrhage.png" alt="Subtypes of hemorrhage" style="width: 70%; text-align: center; display: block; margin-left: auto; margin-right: auto;" />

The dataset is decently sized (~15 gb), so we wouldn’t need to search for similar datasets for training the model. Additionally, the dataset is extensively labelled, making it easy for us to start training the model as soon as possible. We can also apply different types of machine learning algorithms (e.g. CNNs, RNNs, Autoencoders) to try and accomplish the goal of identifying hemorrhages.

---

## Distribution of Report Tasks

- Dataset research - _Micah, Shubham_
- Investigation of research papers - _Davinder_
- Why this dataset - _Shubham, Davinder_
- Jira/Github setup - _Michelle_
- Google Colab setup - _Micah_
- Report finalization - _Shubham_

## Distribution of Project Responsibilities

- Model Creation - _Davinder_
- Transfer Learning - _Michelle, Shubham_
- ML Pipeline - _Micah_
- Analysis & Visualizations - _Shubham_
- Metrics - _Davinder_
- Finalizing & proofreading deliverables (reports, presentations, etc) - _Micah, Michelle_

_This is a tentative list and can be subject to change. See Jira for more granular breakdown of tasks._

---

## Project Links

- [Github](https://github.com/michelleeluo/CSC490)
- [Google Colab](https://colab.research.google.com/drive/1L9u6zy_-MKn0lnWQXHRdtJ-po4zfXnor?usp=sharing)
- [Jira](https://csc490hr.atlassian.net/)
