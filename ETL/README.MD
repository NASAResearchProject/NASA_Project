## ETL & Preprocessing  

This folder houses two files, [ETL](https://github.com/NASAResearchProject/NASA_Project/blob/Andres-branch/ETL/ETL.ipynb) and [Dataset_cleaning](https://github.com/NASAResearchProject/NASA_Project/blob/Andres-branch/ETL/Dataset_cleaning.ipynb).

### ETL  
In the ETL file, we load Sloan Digital Sky Survey DR12, DR14, and DR16 Server Data. The files are simply concatenated together using a list datatype. In the following cells rows with empty values are removed and duplicate rows are removed (in case the same datapoint happened to be in multiple datasets). Some of the feature names in the dataset are keywords in SQL. So we had to rename all of the features by appending an underscore to all column names. We used these methods on the dataset to maintain the structure of the dataset to improve the performance of machine learning models. 

In the final cell is code that will allow someone to export the dataframe into a postgresSQL table.  

## Cleaning Complete Dataset  
In the previous program we merged and transformed the dataset. In this program we preprocess the dataframe as part of the preprocessing step. First all features that have no meaningful statistical value is dropped from the dataframe. This includes features such as ID's, camera specifications, and object location. The remaining features include the class (type of object), which will act as our target feature, metrics of the Thaun-Gunn astronomic magnitude system, and the red shift. To ensure the machine learning model is only being fed numerical data, the types of all features are checked. Lastly, `sklearn.preprocessing`'s library `LabelEncoder` is used on the class feature to quantify the object type id's. 
