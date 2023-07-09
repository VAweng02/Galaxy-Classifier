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
