# nlp-hackathon-2022
This repo hosts utilities for the LiRo NLP Hackathon - held on the 5th of November 2022 in Bucharest, Romania

# nlp-hackathon-2022
This repo hosts utilities for the LiRo NLP Hackathon - held on the 5th of November 2022 in Bucharest, Romania


## Starting from scratch

- from your github useer create a manual fork of the Liro hackathon repository
- next clone that repository to local and add the trained model and a python script to run the model.

		git clone https://github.com/user_name/nlp-hackathon-2022
	cd nlp-hackathon-2022
	mkdir upload
	cd upload
  	mkdir ``team_name``
  	cd ``team_name``
  	mkdir ``task_N``
  	cd ``task_N``

- ``team_name`` is a folder with the name of your team
- ``task_N`` folder represents the folder coresponding to one of the first 5 tasks from the first challenge
- The folder ``task_N`` will contain a ``model.bin``, the binary of the trained model and a ``file.py`` with the following structure:

```python
def <your_model_name>():
  def __init__(self):
    # do here any initializations you require

  def load(self, model_resource_folder):
    # we'll call this code before prediction
    # use this function to load any pretrained model and any other resource, from the given folder path

  def train(self, train_data_file, validation_data_file, model_resource_folder):
    # we'll call this function right after init
    # place here all your training code
    # at the end of training, place all required resources, trained model, etc in the given model_resource_folder

  def predict(self, test_data_file):
    # we'll call this function after the load()
    # use this place to run the prediction
    # the output of this function is a single value, the Pearson correlation on the similarity score column of the test data and the predicted similiarity scores for each pair of texts in the test data.
```
- Install git lfs following the instructions [here](https://git-lfs.github.com/)
- Commit those files to the main LiRo repository:

		git add ``model.bin`` ``file.py``
		git commit -m "lorep ipsum"
		git push origin main

- Create a manual PR to the main LiRO repository
