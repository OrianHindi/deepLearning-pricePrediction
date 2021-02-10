# House price estimation from visual and numeric features

### Abstract <br/>
In this work we present our models of predicting prices for houses in South Korea and USA, California.<br/>
Our first model is simple linear regression in which we insert numeric data about the house, and it returns predicted price.<br/>
The second model for the same data was Neural Network. This model has one hidden layer with 150 neurons.<br/>
In general, the second NN model had better results. For the last part of this project, we needed data with images in order to do CNN, so we found new data about houses in California that has images as well as numeric data, this model's results will be discussed later on this paper. In addition, we  have a third dataset about houses in California. In this dataset there are 4 pictures for every house as well as numeric data.<br/>
### Introduction <br/>
Every person plans on buying a house at some point in his life.<br/>
In our days it gets harder and harder, therefor people are being careful when they try to buy a new house with their budgets. Moreover, it is important to recognize a good and fair deal whether you buy or sell. But how would we know what is a reasonable price for a house? <br/>
There are many ways to answer this question. The first and most known is to consult a real estate appraiser.<br/>
The problem about this way is that the consult alone costs money and most people cannot afford that. Another way is to base our estimation on previous and similar deals. This way is cheap but may not be accurate for the house we are trying to sell or buy. In our project we want to present a new way of estimating a price for a house using deep learning. <br/>
There are many different factors that may affect a house price- starting with size, location, near by facilities, when the house was built and of course selling year.<br/>
In every deal and specifically in a house sale there are two sides that want to be sure they made a good decision and that the price was fair. <br/>

### Related work <br/>
The subject of predicting house prices is known and therefor was studied by many people all over the globe.<br/>
Over the years, the problem of property evaluation was solved in many different ways- from simple linear regression to more complex techniques such as
artificial neural networks and image processing.<br/>
Image analyzing with CNN is also a very known and researched subject, and in our model we combine this two subjects together in order to get good and more
accurate results for the prediction.<br/>

### Required background <br/>
As we said this project presents many different models when each of them works in a different method of data processing.<br/>
The first and most basic model is linear regression. Linear regression is a linear approach to modelling the relationship between a
scalar response and one or more explanatory variables (also known as dependent and independent variables).<br/>
In this first model the numeric features are data about the property and the final calculation outputs the predicted price.<br/>
The second model is neural network. Neural network is a series of algorithms that endeavors to recognize underlying
relationships in a set of data through a process that mimics the way the human brain operates.<br/>
In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. Neural networks can adapt to changing
input, so the network generates the best possible result without needing to redesign the output criteria. Neural network allows us to do more complicated
calculations than linear regression, but outputs predicted price as well. <br/>
In addition, the third model uses image analyzing via CNN- convolutional neural network. The network employs a mathematical operation called convolution. Convolutional networks are a specialized type of neural <br/>
networks that use convolution in place of general matrix multiplication in at least one of their layers. <br/>

### Previous attempts- <br/>
Part 2- in order to build our NN model we encountered some difficulties. One of them was very high values in the loss function since
we dealt with big numbers even the average loss was too big to handle.<br/>
To fix this problem we normalized the data. In other attempt with the neural network we had a dying ReLU. All the weights became zero and the loss function did not
change even after thousands of learning loops. A dying ReLU always outputs the same value for any input. Probably it happened because we had large negative
weights. After normalizing the data, the dying ReLU problem was solved. <br/>

Part 3- as mentioned before we needed to find a new dataset that includes images and numeric features so we could build a new model with CNN for the images and NN for numeric data. After finding such dataset we built and trained a model for it but even after a lot of work to improve our model we got to conclusion that the dataset is not good enough. We suspect it to be since the house images were not in a uniform format and did not reflect the appearance of the house, which caused a very high value for the loss function (around 60 billions). All this led us to look for a better dataset with images in a uniform format in order to avoid earlier mistakes. <br/>
New dataset description can be found in the next paragraph. During building the new model we tried to improve it in several ways.<br/>
The basic model had two hidden layers, the first with 8 neurons, the second with 4 neurons, and each image was resized to 32*32 pixels.<br/>

##### Improvement tries:

* Adding neurons to each hidden layer so that the first has 16 neurons and the second has 8- result: improved model.<br/>                                      
* Adding a third hidden layer- result: slower model with same values.<br/>
* Removing the second hidden layer- result: faster but worse model.<br/>
* Resizing each image to 64* 64 pixels instead of 32*32- result: same values as base model. <br/>
* Adding new features which computed as follows: for every numeric feature we added as a new feature. In total we added 3 new features- result: improved model. <br/>
* Changing optimizer to SGD instead of ADAM- result: worse model. <br/>
* Changing test and train balance- result: worse model. <br/>
* Canceling dropout or decrease dropout precent- result: worse model over fitting. <br/>
 
### DataSet Description-
The collected dataset is composed of 535 sample houses from California State in the United State. Each house is represented by both visual and textual data. The visual data is a set of 4 images for the frontal image of the house, the bedroom, the kitchen, and the bathroom as shown in figure 1. The textual data represent the physical attributes of the house such as the number of bedrooms, the number of bathrooms, the area of the house and the zip code for the place where the house is located. The house price in the dataset ranges from $22,000 to $5,858,000. Table 1 contains some statistical details about our dataset. This dataset was manually collected by H. Ahmed, in 2016.<br/> 

![image](https://user-images.githubusercontent.com/57639675/107544174-a0340500-6bd2-11eb-96f2-377a99f85ae6.png)
![image](https://user-images.githubusercontent.com/57639675/107544217-ac1fc700-6bd2-11eb-8c56-31ddcf40356f.png) <br/> 

### Project description <br/> 
##### Preprocessing- <br/> 
Textual data:<br/> 
-	Numerical features: number of bedrooms, number of bathrooms, house area, and price- feature to be predicted. <br/> 
- Categorical feature- zip code.<br/> 
we normalized numerical features using min-max scalar, and categorical featuresusing one hot encoding. <br/> 
we used np.hstack() function to combine numerical and categorical data.<br/> 
image data: <br/> 
- Every image is normalized and resized to 32*32 pixels, and every 4 images that represent the same house are combined to one image of
size 64*64 pixels. In addition every pixel is divided by 255 in order to normalize the image.<br/> 

##### Models:
First model is multilayer perceptron for the textual data. It has two hidden layers, the first with 8 neurons,  <br/> 
the second with 4 neurons and the activation layer is ReLU.<br/> 
Second model is convolutional neural network for the image data. Every entry(image) is 64*64*3.<br/> 
First layer has 16 kernels of size 3*3 with padding "SAME", activation function ReLU, batch normalization and max pooling (2*2). <br/> 
Second layer has 32 kernels of size 3*3 with padding "SAME", activation function ReLU, batch normalization and max pooling (2*2).<br/> 
Third layer has 64 kernels of size 3*3 with padding "SAME",activation function ReLU, batch normalization and max pooling (2*2).<br/> 
After getting third layer's result we flattened the volume, and added a fully connected layer with 16 neurons, activation ReLU, batch normalization and dropout.
This fifth layer has 4 neurons and activation ReLU. <br/> 

##### Output:
Fully connected model between MLP result and CNN result gives us the final output which is the normalized price prediction. <br/> 
This model has three FC layers with 12,6,1 neurons respectively, first two layers had activation ReLU and third layer had linear activation. <br/> 
We optimize the model output using ADAM optimizer and output this result. <br/> 

###### Model illustration: <br/> 
![image](https://user-images.githubusercontent.com/57639675/107545631-30267e80-6bd4-11eb-9ac9-ca86335b46fe.png)
<br/> 

### Experiments results:
In order to measure the effectivity of the model we created an "average model" (outputs average price for every instance).
average price on train-524,290$ <br/> 
average model test error result- 123,924,830,097 <br/> 
main model test error result- 20,907,451,715 <br/> 
As can be seen our main model has much better result than average model.  <br/> 
![image](https://user-images.githubusercontent.com/57639675/107546429-04f05f00-6bd5-11eb-8997-6932ba2ee7d2.png)
![image](https://user-images.githubusercontent.com/57639675/107546376-f6a24300-6bd4-11eb-92db-4148a425e65c.png)

### Conclusions:
To conclude, in order to get good results for predicting prices using CNN the image data should be in a uniform format. In addition price predicting is more accurate when basing on numeric features rather than images, but combining numeric features as well as images gives better results than numeric features only.

