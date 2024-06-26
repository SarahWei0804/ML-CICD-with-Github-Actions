### Data
The data used in this project is from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Many people died from heart disease. The dataset aims to detect early and manage in advance to prevent the death.

### Description
This project is to implement CI/CD with Github Actions and Hugging Face Space. It is modified from the guidence: [A Beginner's Guide to CI/CD for Machine Learning](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning).
I use random forest classifier as example to predict the possibility to have heart disease. After training, the model will saved and evaluate with CML. CML records the metrics which are accuracy and F1 score.
Finally, the model will deploy to Hugging Face Space.

If a pull request or push action occurred, the code will automatically update with Github Actions. There's no need to manually retrain the model.

The application of Hugging Face is [here](https://huggingface.co/spaces/sarahwei/Heart-Disease-Classification).
