# Galaxy Classifier
This is a final team project in my Computer Vision course that accurately classifies galaxy images.

## Methodology
Inspiration was taken from previous homework assignments from the course, such as the general data pipeline structure and added a few additional features on top of it. In addition, inspiration was also taken from Shibin Paul’s Galaxy Morphology Predictor on Kaggle to initially construct the CNN architecture.

## Preprocessing
Before training the CNN, we preprocessed the galaxy images by cropping them to a resolution of 256x256 to standardize the sizes and reduce complexity. We also converted the images to grayscale, simplifying the input data and reducing the CNN input channels. 
Grayscale was chosen as it focused on extracting features and curvature information from the galaxy images, rather than relying on color. This choice improved feature extraction and reduced computational time significantly.

## Data Splitting
For model evaluation, we divided the dataset into training, validation, and testing sets. The training set was used for training the model, the validation set for hyperparameter tuning and model selection, and the testing set for final evaluation. 
Random partitioning ensured balanced distribution of galaxy classes across the sets, preventing evaluation bias. The split proportions were controlled by variables, allowing flexibility in subset sizes. 
Tuning decisions were made based on analysis of validation and testing results.

## CNN
The CNN expects grayscale images as input with a shape of (N, 1, H, W), where N is the batch size, 1 represents the grayscale channel, and H and W denote the image dimensions. It consists of four convolutional layers (self.conv1 to self.conv4) with varying input/output channels and kernel sizes. 
ReLU activation is applied after each convolutional layer.
Two max pooling layers (self.pool1 and self.pool2) reduce the spatial dimensions of the feature maps. 
Dropout regularization is implemented with two dropout layers (self.drop1 and self.drop2) to prevent overfitting.
The CNN includes one fully connected layer (self.fc1), which produces the output logits for classification based on the flattened input from the previous layers. The output size depends on the number of classes (37 in this case).
The forward() function executes the forward propagation, sequentially passing the input tensor x through the defined layers, applying activations. It returns the final output logits z as a torch.Tensor object.

## Experiments
For the validation of the project, it is best to start with what the model is attempting to describe. The model, given a labeled preprocessed image of an item in space, is tasked with providing answers to the following decision tree questions as a (,37) sized vector (37 for each of the 37 questions).
Although the model doesn’t know what the questions are asking, it is able to predict why some images provide higher answers to some questions than others. So the input data of a labeled image makes sense as the model is determining what the probability for each of the questions is. And a further goal may be to take these probabilities in the 37 length vector and try to categorize the image based on the decision tree.
Some metrics for us to recognize what success is can be by looking at the squared loss of the prediction and the actual label. The loss we have defined can change, but currently a correct categorization of an image is when 35 or more of the 37 probabilities are within 0.1 of the correct answer. This measure is reasonable because when attempting to classify a galaxy, getting more than 35 of the decision tree questions correct means that the classification is very close to what the real answer is, if not fully correct.

## Implementations
We drew inspiration from Shibin Paul's network in the Galaxy Zoo dataset classification competitions. Paul's network featured cycles of convolutional layers, activation functions, and max pooling layers, followed by cycles of dropout layers, dense layers, and activation functions. Based on this architecture, we developed our own model with four convolutional layers, incorporating dropout layers after the third layer. We concluded the network with a fully connected layer. Our implementation utilized PyTorch's network and CNN functions. Subsequently, we focused on tuning hyperparameters such as learning rate, weight decay, loss function, and optimizer selection.

We applied a weight decay of 1 * 10-5 and set the learning rate to 1 * 10-1. For optimization, we utilized Adam Optimization and employed Cross Entropy Loss. Cross Entropy Loss was chosen because it measures the distance between the model's predictions and the actual solutions, aiming to minimize this distance. Despite our efforts in tuning, the model faced challenges in learning. This is likely attributed to the limited dataset size of only 5000 images. With 37 different outputs in the feature vector, training on such a small dataset hindered the model's ability to learn the intricate relationships between the images and all the listed features. Unfortunately, due to time constraints, we couldn't train on a larger dataset. Training on the entire set would have required several hours on GPU settings.

## References
1. “Galaxy Zoo Data,” Galaxyzoo.org. [Online]. Available: https://data.galaxyzoo.org/. [Accessed: 22-Apr-2023].
2. “Intelligent diagramming,” Lucidchart. [Online]. Available: https://www.lucidchart.com/pages/. [Accessed: 22-Apr-2023].
3. Shi, H. "Galaxy Classification with Deep Convolutional Neural Networks." Master's thesis, University of Illinois at Urbana-Champaign, 2016. (34 pages) 
4. A. Ghosh, C.M. Urry, Z. Wang, K. Schawinski, D. Turp, and M.C. Powell, "Galaxy Morphology Classification using Transfer Learning with Pre-trained Convolutional Neural Networks," in Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 112-121. DOI: 10.3847/1538-4357/ab8a47
5. S. Paul, “Galaxy Morphology Predictor.” Galaxy Zoo - The Galaxy Challenge, 2023. [Online]. Available: https://www.kaggle.com/code/shibinjudah/galaxy-morphology-predictor

