# NASA_Project

## Week1

## Communication Protocols  
My group mate Andres and I chose to create a project regarding space related research. The objective is to create and communicate machine learning algorithms that can predict the type of celestial object based on a number of features. There were four aspects we had to cover, the machine learning model, maintaining the repository, database management and data visualizations. Since there are two of us, we split the responsibilities in half. I cover the machine learning and data visualization aspect, and Andres covers the rest. We communicated our expectations and set checkpoints to have done by the weekend. We then met one on one on zoom to discuss where we're at and what trouble we ran into. For instance, some of the column names in the datasets are keywords in SQL, so we had to consider changing the column names. Although we have different responsibilities we worked closely together to make sure one's work was compatible with the other. For instance, I had to ensure the datasets were properly extracted and transformed before Andres could load the data into an SQL database. One thing I appreciate about group work is how one has to constantly communicate their ideas and to make sure everyone is on the same page. It encourages organization. Andres and I are in touch on a daily basis regarding where we are at, what we need assistance with, and what work we are yet to complete to meet our objectives.

### ETL  
We found 4 datasets on Kaggle. The Sloan Digital Sky Survey data offers the public data on space observations. Across the four datasets, we have accumulated over 700 thousand data points. Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar. My team thought this was perfect to build our machine learning models around. We decided to keep the first machine learning model simple, so we only used the three smallest datasets for the first model. The ETL process was split into two parts. The merging and cleaning of the datasets were the first part. After merging the datasets, wenoticed some duplication of rows. We loaded a dataset with over 100 thousand datapoints for the first process. The second part of the ETL process required us to preprocess the complete dataset so that it may be compatible with any machine learning models that we threw at it. First we deleted columns that we considered to be meaningless when it came to predicting a celestial objects class, such as object location and telescope specifications. Then we checked the data types of the remaining columns. The `class_` column defines what type of object the celestial object it in the form of a string, so I used the `LabelEncoder()` method from the `sklearn.preprocessing` library to do so. The resulting dataset looks as so:  
![image](https://user-images.githubusercontent.com/68082808/102816320-d22ba680-439b-11eb-9e09-0444ddb3114d.png)  

We merge datasets [here](https://github.com/NASAResearchProject/NASA_Project/blob/Amir-branch/ETL/ETL.ipynb)  
Our preprocessing code may be found [here](https://github.com/NASAResearchProject/NASA_Project/blob/Amir-branch/ETL/Cleaning%20Complete%20Dataset.ipynb).

### Machine Learning
