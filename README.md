# Machine Learning Classification of Body Part, Imaging Axis, and Intravenous Contrast Enhancement on CT Imaging

This repository contains early demo snippets of the TensorFlow-based code and information related to our research paper on using machine learning for automated classification of CT series. The code may be used for visualizing anonymized public health DICOM data and exploring the initial training viability of our CT classification methods using open public anonymized data. 

Due to potential legal and licensing complexities, the final ResNet PyTorch implementation is not included in this public release.

The published PDF of this paper, as it appears in the Canadian Association of Radiologists' Journal (CARJ), is included in this repository for reference purposes.

In addition, a selection of metrics from the original research is also presented in the readme.md.

## Overview

**Problem:** Manually classifying CT scans for body part, imaging axis, and contrast enhancement is time-consuming and prone to errors due to inconsistencies in DICOM metadata. This hinders dataset curation and can lead to issues in clinical deployment of AI models.

**Solution:** We developed and evaluated deep learning models to automatically classify CT series based on these three crucial parameters.

**Methods:**

*   We used a retrospective dataset of 6955 CT series from our institution, annotated by expert radiologists.
*   The dataset was split into training (70%), validation (20%), and testing (10%) sets.
*   We trained three separate 3D ResNet models for each classification task: body part (16 classes), imaging axis (3 classes), and intravenous contrast (2 classes).
*   External validation was performed using 35,272 series from 7 publicly available datasets.

**Results:**

*   Our models achieved high accuracy on the internal test set: 96.0% for body part, 99.2% for imaging axis, and 97.5% for contrast enhancement.
*   External validation demonstrated strong generalizability with accuracies ranging from 89.7% to 97.8% for body part, 98.6% to 100% for imaging axis, and 87.8% to 98.6% for contrast enhancement.
*   Overall, the models correctly classified all three parameters for 92.7% of series in the internal test set and showed comparable performance on the pooled external validation dataset.

**Conclusion:** Our developed models demonstrate robust performance in automatically identifying key aspects of CT series, which can significantly improve dataset curation workflows and potentially enhance clinical applications.

## Why We Switched from TensorFlow to PyTorch

We initially started our project using TensorFlow but transitioned to PyTorch for several key reasons:

*   **Increased Flexibility and Easier Debugging:** PyTorch's dynamic computational graph allowed for more intuitive debugging and greater flexibility in model development and experimentation. We found it easier to prototype and modify our network architectures with PyTorch.
*   **Strong Community and Active Development:** PyTorch's rapidly growing and active community provides extensive support, readily available tutorials, and a continuous stream of updates and new features. This facilitated faster development and problem-solving.
*   **Pythonic Nature:** PyTorch's more "Pythonic" design felt more natural and easier to work with for our team, leading to a more streamlined development experience.
*   **Research Focus:** PyTorch is often favored in the research community, providing access to cutting-edge models and techniques. This was beneficial for exploring different network architectures and optimization strategies.

While TensorFlow is a powerful framework, PyTorch's strengths in flexibility, debugging, and its strong research community made it a better fit for our project's goals and development style.



## Attribution Statement

This LaTeX document is a formal larger write-up of research originally undertaken by Max Stepanov and Alexis Murari while affiliated with Unity Health and the University of Toronto's Computer Engineering department.