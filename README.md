# LiRo NLP Hackathon
This repo hosts the storage point and utilities for the LiRo NLP Hackathon - held on the 5th of November 2022 in Bucharest, Romania

Below is information for participants on how to **upload** their work at the end of the hackathon for evaluation. 

We have the following tasks :

1. ``task1`` - LiRo Challenge - Task 1 - Named Entity Recognition
2. ``task2`` - LiRo Challenge - Task 2 - Tweet Emotion Detection
3. ``task3`` - LiRo Challenge - Task 3 - Semantic Textual Similarity
4. ``task4`` - LiRo Challenge - Task 4 - Sentence Segmentation
5. ``task5`` - Diac Restore Challenge 

Let's assume you want to upload your code for task1, and your team name is ``MyAwesomeTeam`` 

Please follow these steps: 

### 1. Structure and verify your repository and files

Please work in the following manner: create a folder with your team name (don't include punctuation please), and a subfolder corresponding to your task.

For our example, please create ``MyAwesomeTeam/task1/`` and work in there. 

In the ``task1`` folder please have a file named ``model.py`` that implements all the methods described in the colab for each task. 

These are requirements for the automatic evaluation that will be run at the end of the hackathon. 

### 2. Install GIT LFS 

This is required for task 5 in particular, where we won't have time to train your model from scratch, and you'll have to upload the model itself. Because models are usually >50MB , you'll need to install git lfs to be able to upload larger files. 

There are several tutorials online on how to install git lfs. [Here's a quick tutorial](https://git-lfs.github.com/). 

### 3. Fork this repo

Logged in to GitHub with your user, create a fork of this repo.

Then, clone it locally, and in the ``upload`` folder move the ``MyAwesomeTeam/task1/`` with all your code.

This means that your code will now be in ``nlp-hackathon-2022/upload/MyAwesomeTeam/task1/``

If you have large files, init git lfs at this point, and, in the command line, run ``git lfs track <your-big-file-here>

Then run, from ``nlp-hackathon-2022/upload/MyAwesomeTeam/task1/`` :

```bash
git add .
git commit -m "<team name here>"
git push origin main
```

this should upload all your files to your repo fork!

### 4. Create a Pull Request from your commit to the main repo

This is the last step, and done online from github.com from your repo fork. Click on the new commit in your fork and click Create a new Pull Request to the LiroBenchmark/nlp-hackathon-2022/ repo.
