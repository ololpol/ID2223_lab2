# Air Quality Prediction Pipeline with Hopsworks

This project finetunes LLama-3.2-1B-Instruct on the FineTome100K dataset. The model is trained using **Unsloth** on google colab, and is hosted at **Hugging Face**. Hugging Face spaces is used to make the model avaliable with the help of **Gradio**. The finetuned model is compared to the original model using user prompts or sample promts from the training dataset.

The finetuned model is avaliable at https://huggingface.co/oloflil/model

A hugging face space provinding a access point to use the model for inference is accessible at https://huggingface.co/spaces/oloflil/ID2223-lab2

The app.py file can also be ran to host a local gradio space using the finetuned model:

python3 -m venv ./.venv

source .venv/bin/activate

pip install -r requirements.txt

python app.py

# Improving model performance.

## Model-centric approach.
There are a few sections in the notebook that perform hyperparameter optimization in order to attempt to find the best hyperparameters. However, limitations in available compute, as well as the large dimensinoality limits the number of hyperparameter configurations that can be tested. Given more time, several more configurations could be tested when performing successive halving, which could lead to better results.

Attempting to finetune larger model could also yield a better performance, but more parameters slows down the (already quite slow) inference and training.

## Data-centric approach
What data sources give a better performance depends on what the trained model is used for. Here the model is finetuned on the FineTome100K dataset that contains questions, and answers to thoose questions. In order to get better result it may be beneficial to find additional Question answering datasets. Some examples of such datasets are:

- SQuAD(stanford question answering dataset) containing questions about various wikipedia articles.
- Googles natural questions dataset.
- TriviaQA that contains several trivia Q&A pairs.


# Usage instructions(for replicating the project)

    
## Train the model using colab

Upload the ID2223_lab.ipynp file to google colab
Replace the HF_USER and HF_TOKEN values to match your hugging face account

Run the notebook train the model.

The file trains the model and stores it as gguf at google drive.
It is also possible to directly save the trained model to hugging face.


## Upload the model to Hugging Face

Download the file model-1b-Q8_0.gguf from google drive and upload it to a model directory at hugging face.

Replace the value of MODEL_1 in app.py to the directory where the model is located at hugging face.

## Run the Gradio app locally

python3 -m venv ./.venv

source .venv/bin/activate

pip install -r requirements.txt

python app.py

## Run the Gradio app at hugging face spaces

Upload requirements.txt and app.py to a hugging faces space. 
Additionally, create the file README.md and add the following to it:

title: ID2223 Lab2

emoji: 💬

colorFrom: yellow

colorTo: purple

sdk: gradio

sdk_version: 5.42.0

app_file: app.py

pinned: false

hf_oauth: true

hf_oauth_scopes:
  - inference-api



