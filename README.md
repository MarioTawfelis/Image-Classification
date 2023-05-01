# Image Classification

This is an end-to-end image classification porject to identify key female leaders of the 21st century.

My goal for this project was to build an end-to-end ML project rather than an emphasis on model accuracy or incorporating state of the art ML models. This project can classify 4 female leaders: Ai-Jen Poo, Beyonce, Nancy Pelosi and Oprah Winfrey.

This project involved several tasks including:
- Data Collection: building a dataset by gathering images of key female leaders of the 21st century utilising different Python packages such as Selenium and Requests.
- Data Cleaning: this process was divided into two tasks: 1) identifying faces using Python's OpenCV package and an algorithm called Haar cascade, and 2) discarding images that do not contain visible face or obstructed eyes as well as faces of individuals we are not interested in identifying. 
- Feature Engineering: using cropped images of the individuals we are interested in classifying, wavelet transform was used to analyze and convert images into distinctive meaningful features. Raw image vertically stacked with the wavelet transformed image are used represent the features or independent variables and the target variables or labels are the names of the individuals that we are trying to classify.
- Model training: I have used three models Logistic Regression, SVMs, and Regression Trees. Using GridSearchCV to hypertune the parameters for each model, I found that Logistic Regression and SVMs performed better in terms of model accuracy with Logistic Regression being the best performing model. Train, validation and test dataset splits were used for training, tuning hyperparameters and evaluation, respectively. Using the Seaborn package, I built a confusion matrix to visualise classification errors (true positives and false positives, true negatives, false negatives). Finally, the finalised model was exported to a pickle file using the joblib package.
- Frontend/UI: building a Python Flask server to serve as the frontend for my project. The UI is fairly simple, it contains an input field where the user can drag & drop or upload an image that he/she wishes to classify and the output is the name of the classified person along with similarity scores with other individuals. An error message shows up if the model could not classify the input image.
- Deployment: the final project was then hosted on an AWS EC2 instance and served and proxy reversed using NGINX. SFTP was used to copy code to the EC2 instance.
