# Bayesian-Machine-Learning-in-Breast-Cancer-Diagnosis

The aim of this project is to apply Bayesian Machine Learning Algorithm to predict the diagnosis condition of the patients based on the sample collected from them.Here different measurements of the cell nuclei are the information that we provide to the machine to gain experience during training.

## Models were Fitted on:
-1.Transformed Data (Centering,Scaling,Yeo-Johnson Transformation,Spatial Sign Transformation)

-2.Raw Data

##  Variables Selection:
-1.Least Absolute Shrinikage and Selection Operator (LASSO) 

-2.Recursive Feature Elimination (RFE)

## Density Estimation:
-1. Probability Density of Normal Distribution

-2. Non-Parametric Guassian Kernel Density 

### Thus, in total eight Bayesian Machine Learning Models were fitted. The model performance will be discussed in details in a separate file.

---
### Data Description
The data used in this project is a Breast cancer diagnosis dataset. Using the digital image of Fine Needle Aspirates(FNA) from breast mass of 569 individual, different measurement was made for each cell nucleus. In the dataset, total 32 variables and 569 observations were counted.
A short description of the Variables is as follow:

Variable 1: ID number

Variable 2: Diagnosis (M = malignant, B = benign) 

Variable 3-32: Ten real-valued features were measured for each cell nucleus: 

a) radius (mean of distances from the center to points on the perimeter) 

b) texture (standard deviation of gray-scale values) 

c) perimeter 

d) area 

e) smoothness (local variation in radius lengths) 

f) compactness (perimeter^2 / area - 1.0) 

g) concavity (severity of concave portions of the contour) 

h) concave points (number of concave portions of the contour) 

i) symmetry 

j) fractal dimension ("coastline approximation" - 1)

For each of these 10 different measurements information about mean, standard deviation and the worst condition for every individual was assessed and stored as independent variables. Note that, the size of cell nuclei is associated with the breast cancer-causing genes (BRCA1, BRCA2, ATM, BRIP1, CDH1, CHEK2 and several others).

Variable 1 indicates the ID number of individuals. Note that, in fitting the Bayesian Machine Learning model, variable #02 i.e. Diagnosis was treated as the dependent variable which contains two classes “B” and “M”. ”B” stands for Benign and it indicates that the tumor is not cancerous. These types of tumor do not harm the surrounding tissues. On the other hand, ”M” stands for Malignant tumor and it is cancerous. The malignant tumor grows uncontrollably and it badly harms the surrounding tissues.
This dataset was collected from University of California, Irvine (UCI) Machine Learning Repository. Further details of the dataset and list of scholarly papers that used this dataset can be found at the following link:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

---

### Contributors
- Mohammad Ehtasham Billah <mymun.stat@gmail.com>

---

### Licence & Copyright
© Mohammad Ehtasham Billah , Örebro University, Sweden

