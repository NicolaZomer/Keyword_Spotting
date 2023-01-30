# Speech command recognition (Keyword Spotting)
In this project we use the [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands), which contains short (one-second long) audio clips of English commands, stored as audio files in the WAV format. More in detail, the version 0.02 of the dataset contains 105.829 utterances of 35 short words, by thousands of different people. It was released on April 11th 2018 under [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and collected using crowdsourcing, through [AIY](https://aiyprojects.withgoogle.com/) by Google. Some of these words are "yes", "no", "up", "down", "left", "right", "on", "off", "stop" and "go".

This project is developed as a final project of the course  [Human Data Analytics](https://en.didattica.unipd.it/off/2022/LM/IN/IN2371/004PD/INP9087860/N0).

# Notebooks

- [Data Analysis And Preprocessing Inspection](./notebooks/01_data_analysis_and_preprocessing_inspection.ipynb) <br>
    This notebook takes care of loading and preparing the dataset, splitting it into training, validation, and testing sets. It also provides some information about the dataset with plots. It also introduces the functions used to pre-process the data (for example adding noise).

- [Keyword Spotting: general training notebook](./notebooks/02_keyword_spotting_intro.ipynb) <br>
    This notebook defines the generale training and testing pipeline, giving some information about the validation metrics used. The training is performed using our baseline model `cnn-one-fpool3`, taken from [Arik17].

- [Bayesian Optimization and Feature Comparison with CNN](./notebooks/03_cnn_bo_fc.ipynb) <br>
    This notebooks is used to train our custom CNN models. With the first of these models we perform a Bayesian optimization, and we use it for inspecting the importance of dropout and batch normalization, realizing a feature comparison and studying the effect of data augmentation on the training set.

- [Keyword Spotting: ResNet architecture and Triplet Loss implementation](./notebooks/04_resnet.ipynb) <br>
    In this notebook we play with ResNet models for the keyword spotting task. We start by implementing a simple ResNet architecture inspired by [Tang18] and then, motivated by [Vygon21], we modify such model and we train it to get a meaningful embedded representation of the input signals. We finally use k-NN to perform the classification task on these intermediate representations.

- [Keyword Spotting: a neural attention model for speech command recognition](./notebooks/05_crnn_with_attention.ipynb) <br>
    This notebook implements an attention model for speech command recognition. It is obtained as a modification of a [Demo notebook](https://github.com/douglas125/SpeechCmdRecognition/blob/master/Speech_Recog_Demo.ipynb) prepared by the authors of the paper [A neural attention model for speech command recognition](https://arxiv.org/abs/1808.08929).

- [Keyword Spotting: Conformer](./notebooks/06_conformer_bo.ipynb) <br>
    In this notebook, thanks to the library `audio_classification_models`, we implement a baseline Conformer architecture inspired by [Gulati20]. This model combines **Convolutional Neural Networks** and **Transformers** to get the best of both worlds by modeling both local and global features of an audio sequence in a parameter-efficient way. In detail, we use only one Conformer block in order to reduce the number of model parameters. Moreover, we perform hyperparameter tuning by means of Bayesian optimization in order to find, among the models with less than 2M parameters, the one that leads to the best accuracy.

- [Keyword Spotting: GAN-based classification](./notebooks/07_conditional_dcgan.ipynb) <br>
    In this notebook we try to implement a GAN-based classifier inspired by the paper [GAN-based Data Generation for Speech Emotion Recognition](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/2898.pdf). Unfortunately, to date we have not been able to figure out how to properly train the generator and discriminator in this specific case. As a result, we cannot currently test this approach.

# Utils

- [Models Utils](./utils/models_utils.py)
- [Plot Utils](./utils/plot_utils.py)
- [Processing Utils](./utils/preprocessing_utils.py)

# Demo App 
In this repository you can find a demo application that can be run as a python script with `python demo_ks.py`. It allows you to select the model you want to use and, when started, it detects the words in the Speech Commands Dataset through the microphone (or any chosen input device).

You can also find a [notebook](./play_commands.ipynb) that can be used to play some commands from the dataset, in order to test such application with non-real-time signals.

# Collaborators 
- Daniele Ninni <br>
- Nicola Zomer