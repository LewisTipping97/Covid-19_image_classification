# Covid-19_image_classification
I was given images from a covid-19 CT scan, the task was to classify images as covid positive/negative. To do this two techniques were used. The first was logistic regression, this incorportated LASSO as a form regularisation. Secondly, XGBoost was used. The XGBoost model was deemed the more accurate.

Following this, some data regarding the images was provided which had been passed through a neural network. This data was reduced via PCA and then Logistic regression was once again applied. In this task there was a deeper look at the regularisation of logistic regression. The data was then clustered using a Gaussian Mixed model.
