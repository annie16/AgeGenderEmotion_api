# keras-js-facial_expression
Dataset :
Train Set contains 36517 images collected from FER training set, SFEW training set , KDEF nd CK+ sets

Test  on SFEW: 
59.2%


*********************************************************************************************************


Running Train.py gives training accuracy and model can be saved for testing, it is stored as baseline.h5

test.py loads basline.h5 model and tests the model on the testing data and ouputs accuracy 

dataset.py converts dataset into numpy array and labels as one hot encoding
