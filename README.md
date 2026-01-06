# MRI Alzheimer's Classifier
Task: Multi-class MRI classification (Non Demented, Very Mild, Mild, Moderate)

# 1. Motivation and problem statement
While Alzheimer’s is progressive, it often isn’t caught early, severely limiting treatment options
and reducing the time patients and families have to plan care. And with real-world clinics
overloaded, the biggest risk is missing a case that should be flagged for follow-up. So I wanted
to build an AI model that focuses on safety-first screening. (aka a model that’s optimized to
reduce missed positives and stays reliable even when MRI scans vary in quality, contrast, and
number)

# 2. Dataset source
Public Kaggle dataset: Augmented Alzheimer MRI Dataset | Kaggle (uraninjo) [\[1\]](#hi4). 
The dataset contains four labeled classes: Moderately Demented, Mildly Demented, Very Mildly
Demented, and Non Demented. I chose to use the original dataset over the augmented dataset so
I could apply augmentation myself and better simulate real-world variability during training.

# 3. Model Details
Model: InceptionReseNetV2 as the backbone. This architecture combines
Inception-style multi-scale feature extraction with residual connections, which is useful for
capturing subtle anatomical differences in medical imaging.
Preprocessing: Each MRI image is resized to a consistent input shape, converted from grayscale
to 3-channel RGB, and normalized using the backbone’s preprocess_input function.
Data augmentation and shuffling: I set a 70-15-15 training, validation, and testing split. I used
minimal rotation, translation and zoom to apply augmentation while also maintaining medical
accuracy.
Class imbalance handling and re-weighting: The dataset is imbalanced (for example, the
Moderate class only has 10 images while Non-Demented has thousands). Initial training
produced high accuracy (~94%) but an outrageous number of false negatives. I adopted the
mindset of being safe rather than sorry, meaning that I needed to focus on sacrificing false
positives for false negatives. In real life, having mild or very mild cases ignored is a **huge** issue.

# 4. Evaluation
I learned that accuracy wasn’t enough when it came to datasets with high variation with the
number of dataset class sizes. So I also decided to use precision and recall. We value precision
because, while we would rather have false positives instead of false negatives, we also don’t
want to waste too many resources on finding out they don’t have Alzheimer’s. The left contains
the confusion matrix with just weighting while the right is the updated with reweighting results.
<br/>
![Alt text](/mri%20classifier%20images/initial%20testing%20results.png)
![Alt text](/mri%20classifier%20images/results%20after%20weight%20redistribution.png)

# 5. Issues Faced
I initially used the tf.keras.utils.image_dataset_from_directory function to split the training,
validation and testing images, resulting in a 40-50% accuracy rate. Clearly, something was
wrong. I discovered that because of the huge imbalance between the images of each of the
classes, I wasn’t getting a good representation from each class causing inconsistent or even
irrelevant results. So I redid the whole process by reading it all at first and then splitting into
datasets by class.
The large variation between the number of images also led me to learn about reweighting the
weights of each class (aka how important the model takes specific classes when it came to
training). More details can be found in the google colab.

# 6. Limitations and Next Steps
This is still a prototype trained on a single public dataset; clinical deployment would require
more rigorous testing, a better, more specially curated backbone for MRI images, and external
validation across sites/scanners. More interpretability is also an important next step to take. For
example, adding heatmaps like GradCAM, and a user-determined threshold would help doctors
personalize the diagnostic process and better understand what the model is doing.

# References
[1] <a id="hi1"></a> Kaggle (uraninjo). Augmented Alzheimer MRI Dataset. https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
