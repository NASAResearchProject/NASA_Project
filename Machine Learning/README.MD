## Machine Learning Models

### K-Nearest Neighbors  
For our first model I chose the supervised algorithm known as K-Nearest Neighbors. The algorithm is simple and elegant. It also required no training step, and is very easy to implement. After the preprocessing step, we had a total of 6 independent features, a relatively small number compared to those of other datasets. This means we don't have to concern ourselves with the curse of dimnesionality, because as the number of variables grow, the K-NN algorithm struggles to predict the output of new data. All of these characteristics of the algorithm and our dataset make choosing the K-NN algorithm an easy choice, and it paid off. Our model was able to predict the class of a celestial object with 95% accuracy.

A scatter matrix depicts a covariance matrix of the preprocessed multivariate feature set. The `class_` feature is designated as the target variable because that is the descriptor that we are trying predicat based on the other independent features. The `sklearn` machine learning library is used to split the dataset into a training set and a test set. The size of the test set is arbitrarily set to be 30% of the whole dataset. The `KNeighborsClassifier` library is used the fit a K-Nearest Neighbors model, and make predictions on the test set. The model performed with an accuracy of 0.954, and we are satisfied with this level of accuracy.

Moving forward I would like to consider standardizing all features so that they're homogeneous. K-NN works best when features have the same scale. All columns except for redshift_ seem to be homogeneous, but nonetheless it is worth tinkering with. I would like to add the fourth dataset to the dataframe as well, because the more datapoints the more data the machine has to learn with. This will allow me to use more robust and complex models such as a neural network.

Our next steps are to run a K-Nearest Neighbors on a number of preprocessed datasets, each with a single unique feature removed. This experiment may shed light on which features offer the most significance with regards to the models performance. 

---
## Scaling Features
The variables of the data set are of different scales i.e. one variable is in thousandths and others in the 10s. For e.g. in our data set redshift_ is having values in thousandths and the other features are of a different scale. Since the data in these variables are of different scales, it is tough to compare these variables. We will convert variables with different scales of measurements into a single scale. StandardScaler normalizes the data using the formula (x-mean)/standard deviation. We will be doing this only for the numerical variables.  

So I entered the [Dataset_cleaning](https://github.com/NASAResearchProject/NASA_Project/blob/main/ETL/Dataset_cleaning.ipynb) file and added the following code to scale all columns.  
```
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale

features_to_scale = ['u_', 'g_', 'r_', 'i_', 'z_', 'redshift_']

for feature in features_to_scale:
    DR_df[feature] = std_scale.fit_transform(DR_df[[feature]])
```  
This resulted in an increase in KNN model accuracy, 96.85 percent.

---

## Balancing Classes with SMOTE  
According to the [Dataset_cleaning](https://github.com/NASAResearchProject/NASA_Project/blob/main/ETL/Dataset_cleaning.ipynb) file, there exists a class imbalance in the target feature `class_`.  
![image](https://user-images.githubusercontent.com/68082808/103785920-c382f780-5009-11eb-81d4-5067d57fe778.png)  

In order to overcome this class imbalance we will use SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority classes. This is a type of data augmentation that synthesizes new samples from existing ones. SMOTE selects samples in the minority class that are close by a KNN algorithm to draw lines between the classes. With this procedure, we can create as many synthetic samples as needed, making SMOTE perfect for datasets of all sizes. Before we apply SMOTE to the dataset, we will need to massage the data. We use `MinMaxScaler` from `scikit-learn` to scale columns that have values greater than 1 to [0, 1] range. I include the if statement in the for loop to ensure that the target variable, `class_`, is not affected. The following code is applied:  
```
from sklearn.preprocessing import MinMaxScaler

# Scale only columns that have values greater than 1
to_scale = [col for col in DR_df.columns if DR_df[col].max() > 1]
mms = MinMaxScaler()
scaled = mms.fit_transform(DR_df[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

# Replace original columns with scaled ones
for col in scaled:
    if(col != 'class_'):
        DR_df[col] = scaled[col]
```  
Now that the dataset is massaged properly, we can now import `SMOTE` from the `imblearn.over_sampling` library, and run the following code:  
```
from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state=42)

X = DR_df.drop('class_', axis=1) # independent features
y = DR_df['class_'] # Target

X_sm, y_sm = sm.fit_resample(X, y)

print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100
```
Note that I redefine the X and y variables as they have been redefined in the previous step. Afterwhich I rerun the for loop to discover the optimal value for k, and then fit the model on the newly balanced dataset. After assessing the model, we find the model accuracy has increased to 97.6 percent. The following classification report follows:  
![image](https://user-images.githubusercontent.com/68082808/103791844-f2509c00-5010-11eb-9859-08ddabf9b555.png)

I follow the same procedure for the other KNN model where the `redshift_` feature is dropped. Applying SMOTE to the dataset in the [KNN_Dropped_redshift](https://github.com/NASAResearchProject/NASA_Project/blob/Amir-branch/Machine%20Learning/K_Nearest_Neighbors.ipynb) increases the models accuracy from 87.6 percent to 90.0 percent.
