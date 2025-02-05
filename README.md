## Wildlife Species Classification from Camera Trap Images

### Problem Statement

Automated surveillance systems called camera traps play an important role in conservation reserach, as it allows researchers to passively record the behavior of animal species in their natural habitat without significant human interference. This data can be used, for example, to study the effects of climate change or wildlife encroachment, by tracking shifts in animal habitats over time. While these systems make data collection easy, it becomes difficult or impossible to analyze the vast quantity of data that they generate. Machine learning systems can help researchers sift through these camera trap images and identify the most important captures by detecting, classifying, and localizing species, among other tasks.

In this project, our aim is to develop a model for classifying such camera trap images into one of eight categories, corresponding to seven animal species, and a single category for images in which no animal is present.

The original task is described in the [DrivenData competition](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

### Data

In the original challenge, we are provided with three CSV files:
- training_set_features.csv
- training_set_labels.csv
- test_set_features.csv

Each row in the above datasets represents an individual image captured from a camera trap located in a specified "site", or zone of the conservation area. The files `training_set_features.csv` and `test_set_features.csv` have rows consisting of an image id, along with the path of the corresponding image in the `train_features` and `test_features` folders, and the site id of the site where the image was captured. The file `training_set_labels.csv` have rows consisting of an image id, as well as eight binary columns corresponding to the one-hot representation of the image label (i.e. what animal is present in the image).

![image samples](img/image_samples.png)

Since the challenge does not provide us with the test set labels, we treat the provided training set as the full data set. Thus, we split the provided training set into training and test sets to train and validate our model, as well to report the final out-of-sample performance on the test set.

### Evaluation Metric and Data Splits

Although the official competition measures classification performance using cross-entropy loss, we use accuracy instead. A quick inspection of the label distribution does not suggest any severe class imbalance. Most labels are roughly within 2 percentage points of 12% of the overall data. The biggest outlier is the `hog` label, which encompasses roughly 6% of the data.

![label distribution](img/label_distribution.png)

We proceed to split the training set into a train, validation, and test set according to a 70-15-15 split. These splits are stratified, so as to maintain the label proportions. More detailed exploratory analysis of the data can be found in the `exploratory_data_analysis.ipynb` notebook.

### Model

The amount of training data available after performing the train-validation-test split is roughly 12k images. This is a rather small dataset for image classification tasks, so we are restriced to using smaller models. This is because a network with too many parameters could easily overfit the training data. Our model is a fairly small convolutional neural network (CNN) model.

Initially, we resize all images to dimension `(150, 150, 3)` and rescale pixel values to be in the interval `[0, 1]` by applying a scale factor of `1/255`. Then we apply 5 convolutional layers with an increasing number of filters, while the max pooling layers in between each convolutional layer halves the input dimensionality at each step. After the final convolutional layer, we have 256 5x5 feature maps, which are then flattened to a rank-1 tensor and passed through 3 Dense layers, with Dropout layers in between. The final Dense layer maps to $\mathbb{R}^8$, and we apply a softmax activation to get a predicted probability distribution over the labels.

<p align='center' width='100%'>
<img src='img/small_cnn.png' width='40%'>
</p>

The small "from-scratch" CNN model achieves an accuracy of 85% on the held out test set. The full process of hyperparameter tuning, training, and model selection can be seen in the `small_cnn_model.ipynb` notebook.

### Install Dependencies

The dependencies for this project can be installed using `pipenv`. First, install `pipenv` using `pip`:

```
pip install pipenv
```

Then, the dependencies can be installed by first changing directory into the root folder of this project, and then running the following:

```
pipenv install
```

Note that this will create a virtual environment for this project. The virtual environment can be activated by running:

```
pipenv shell
```

### Downloading the Dataset and Creating Directories by Split

Since the data directory is too large to store on GitHub, there are two scripts that will do the following: download the compressed dataset, split the data according to the chosen train-validation-test proportions, and create the hierarchical data directory structure expected by the `image_dataset_from_directory` function from Keras.

First, run the following script to download the compressed dataset:

```
python scripts/data/download_dataset.py
```

Then, run the following script to create the directory structure:

```
python scripts/data/create_directories_by_split.py
```

### Training the Model

The `bin/custom` directories already contain the best models (by validation loss) for each run of hyperparameter tuning. The best overall model is `custom_dropout_0.5_100_0.846_0.521.keras`, and this is the one that we put into production.

In order to train this model (with the same hyperparameter values), run the following script:

```
python scripts/train.py
```

This script will save the best performing model (with lowest validation loss) as `best_model.keras` in the `bin/custom` folder.

### Deploying the Model

There are two approaches we took for deploying the model, both locally and on Amazon Web Services (AWS). The primary difference between the approaches is that the first approach uses the full TensorFlow library in deployment, and the second approach uses TensorFlow Serving, which is better optimized for deployment scenarios for two main reasons:
1. It is a lightweight system specifically made for serving trained models, and thus, discards all features of TensorFlow besides inference capabilities. It also has a highly optimized, low-level implementation in C++.
2. It supports serving models via gRPC, which is a data format that encodes data using a more efficient binary representation. This makes it more efficient than JSON, which is based on key-value pairs and high-level data structures such as strings, numbers, and objects. The catch is that deployment involves more moving parts and complex logic.

### Deploying with Pure TensorFlow, AWS Lambda, and API Gateway

AWS Lambda is a serverless compute service, where we can run programs without explicitly provisioning or managing servers. It automatically scales up and down based on traffic to our program.

#### Local Deployment

AWS provides a Lambda Python base image that makes it easy to deploy code Python code to Lambda. We simply extend the base image in `Dockerfile` where we copy over the trained model, as well as a `lambda_function.py` script implementing a `lambda_handler` function. This function is able to receive POST requests and then compute the prediction for the corresponding input. We set this function to be invoked upon the container's start up by overriding the `CMD` argument.

By default, the Lambda image listens for requests on port 8080, at the endpoint `/2015-03-31/functions/function/invocations`. 

Now, we simply build the Docker image:
```
docker build -t species-model .
```
And run the built image as a container, forward our host port 8080 to the conatainer port 8080, where Lambda is listening for requests:
```
docker run -it --rm species-model:latest -p 8080:8080
```

With just this much, we've essentially deployed our model locally with Docker and TensorFlow. Now we can simply make a POST request to the endpoint provided in the Lambda documentation. An test script to do this is provided at `scripts/prod/test/test_lambda.py`. 

#### Upload Docker Image to AWS Elastic Container Registry (AWS ECR)

From this point, it is fairly straightforward to deploy our containerized application to AWS Lambda. We need to install `awscli`, the AWS CLI tool, as follows:
```
pip install awscli
```
Once the AWS CLI tool is configured, the first step is to create an image repository in AWS Elastic Container Registry (ECR). We do this as follows:
```
aws ecr create-repository --repository-name species-images
```
This command will create the repository and output some information, including the URI of the repository, which will look as follows:
```
<account_name>.dkr.ecr.<region>.amazonaws.com/<repository_name>
```
Now, we need to tag our local built Docker image `species-model:latest` with a remote URI, which will have the repository URI as a prefix. For example:
```
PREFIX=${ACCOUNT_NAME}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}
REMOTE_URI=${PREFIX}:species-model-custom-001
```
Now, we need to push the image; but first, we need to log in to AWS ECR:
```
$(aws ecr get-login --no-include-email)
docker push ${REMOTE_URI}
```

#### Create Lambda Function from Image

Now, we need to create a Lambda function from our Lambda image. We simply create a new function in the AWS Lambda console, select "Container image", give the function a name, and point it to our uploaded Docker image. Leave all other settings as default.

#### Expose the Lambda Function via REST API (API Gateway)

Now that we've uploaded our Lambda image to ECR and created a corresponding Lambda function, we need to expose the Lambda function as a web service. We will do this via API gateway, which is another AWS service for creating and exposing REST APIs.

We go to the API Gateway web console, create a new REST API, and assign the API a name. Then we create a resource, and name it `/predict`. We then create a POST method for this resource and select `Lambda Function` as the integration, specifying our previously created Lambda function. Finally, we just need to deploy our API, and API Gateway provides us with an invocation URL.

Whereas we previously did port-forwarding to make requests to our Docker image, we can now use this invocation URL. Again, the same test script `scripts/prod/test/test_lambda.py` demonstrates how we can request this endpoint. Thus, we've successfully deployed our model as a web service using TensorFlow, AWS Lambda, and API Gateway!

### Deploying with TensorFlow Serving

#### Deploying Locally with Docker-Compose

#### Deploying Locally with Kubernetes

#### Deploying to the Cloud (AWS Elastic Kubernetes Service)