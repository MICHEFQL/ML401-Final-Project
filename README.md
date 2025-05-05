# "Is this Pawsitive?": Sentiment Analysis of Cat Memes with Vision Models
## Overview
In this project, we examine vision models' ability to analyze sentiment in cat memes. Sentiment analysis of cat memes could be challenging for current models, because 1) features in cat memes might not translate directly into labels, and 2) the sentiment of a cat meme is often context-dependent, making it difficult to even assign a label. We finetuned and evaluated ViT, a transformer model, and MemSem (Pranesh and Shekhar, 2020), a CNN model, on a dataset of manually labeled cat memes. We found that ViT's performance improved and MemSem's performance worsened after training loss converged during finetuning, while human labelers outperformed both models by a large margin. Our results suggest that while ViT demonstrates reasonable generalizability in sentiment analysis of cat memes, it still struggles with subjective and context-dependent cases, such as identifying positive cat memes.

## Replication instructions
1. Clone GitHub repository
2. ViT: Load evaluate.ipynb in Google Colab and run the script to evaluate ViT model
3. MemSem:
    1. MemSem is adapted from an existing codebase from a literature (Pranesh and Shekhar).
    2. The original dataset which contains 2282 cat memes in png format without text can be previewed and downloaded at [282 cat memes in png format without text](https://www.kaggle.com/datasets/michefqli/401-data-v-anzi).
    3. We have labeled 500 memes which can be found in dataset.csv; the images are stored in the Dataset folder by their labels (done with datagen_1.py).
    4. In the MemSem folder, the model.py has been adapted to work with this dataset (no text input), user should not need to modify anything in model.py and preprocessing.py to replicate our results except one line in predict() function in model.py.
    5. The dataset is split into 450 images for training which are further split into training set and validation set in the training process and 50 images for testing.
    6. To train the model, first replace the value of image_folder in train.py with the local path for the downloaded data folder from Kaggle or directly link to Kaggle. Then running train.py will train the model with our dataset and automatically save the updated weights. (The random seed was set to 42 for consistent results) The initial metrics on the test set before training will be printed out in the console at the bottom. To get the final metrics on the test set after finetuning, uncomment line 208 (model.load_weights('./MemSem.weights.h5') in model.py in the predict function which will make predictions with the trained model, then run model.py and line 46 - 64 (code for printing the test set metrics) in train.py again. 

## Future Directions
Future work can extend and improve this project from two aspects. From a training perspective, the models would likely benefit from a larger training dataset, particularly with increased representation of the positive class, where ViT currently underperforms. We currently have only labeled 500 cat memes, where only 400 can be used for training. From a data perspective, labeling reliability could be improved by having each image annotated by multiple individuals, followed by cross-validation to ensure consistency. Recruiting labelers from a more diverse population would also be helpful, as people's judgments on cat memes may be influenced by their online identities and cultural backgrounds.

## Contributions
Anzi:
1. Brainstormed ideas for the project, conducted literature review, and wrote ~50% of the proposal (3 hrs)
2. Preprocessed images, created dataset and labeled ~400 cat memes (5 hrs)
3. Created pipeline to finetune and evaluate ViT (5 hrs)
4. Analyzed results, wrote 4 sections of the poster and 2+0.5+0.5 sections of the artifact (3 hrs)

Michelle: 
