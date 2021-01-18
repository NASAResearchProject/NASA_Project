## ETL & Preprocessing  

This folder houses one file, [Dataset_cleaning](https://github.com/NASAResearchProject/NASA_Project/blob/Andres-branch/ETL/Dataset_cleaning.ipynb).

### Dataset_cleaning  
In the Dataset_cleaning file, we load Sloan Digital Sky Survey DR12, DR16, and DR16Plus Server Data. The files are cleaned by dleleting the null rows and repeated rows. Then we change the columns names so we do not have problems with column names that are key words in SQL; so we had to rename all of the features by appending an underscore to all column names. Moreover, we simply concatenated the files together using a list datatype. Furthermore, we deleated any repeated rows. Then we separated the data into two different data frames by dividing the columns into the two new data frames that will share the objid_ column. Lastly we exxported these two data frames into csv files and exported these two data frames into a postgreSQL database. We used these methods on the dataset to maintain the structure of the dataset to improve the performance of machine learning models.  
