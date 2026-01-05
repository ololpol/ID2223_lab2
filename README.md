# Parameter Efficient Fine-Tuning (PEFT) of a Large Language Model on a GPU

This project finetunes LLama-3.2-1B-Instruct on the FineTome100K dataset. The model is trained using **Unsloth** on google colab, and is hosted at **Hugging Face**. Hugging Face spaces is used to make the model avaliable with the help of **Gradio**. The finetuned model is compared to the original model using user prompts or sample promts from the training dataset.

The finetuned model is avaliable at https://huggingface.co/oloflil/model

A hugging face space provinding a access point to use the model for inference is accessible at https://huggingface.co/spaces/oloflil/ID2223-lab2

The app.py file can also be ran to host a local gradio space using the finetuned model:

python3 -m venv ./.venv

source .venv/bin/activate

pip install -r requirements.txt

python app.py



# Evaluation
In order to evaluate the finetuned model, the finetuned and original model are compared qualitatively by inspecting the responses on a sample of questions from the FineTome100K dataset. The following question/answer pairs were used. Additionally, there exists support for asking questions directly and getting responses from the two finetuned models.


QA_PAIRS = [
    ("How do astronomers determine the original wavelength of light emitted by a celestial body at rest, which is necessary for measuring its speed using the Doppler effect?", 
     "Astronomers make use of the unique spectral fingerprints of elements found in stars. These elements emit and absorb light at specific, known wavelengths, forming an absorption spectrum. By analyzing the light received from distant stars and comparing it to the laboratory-measured spectra of these elements, astronomers can identify the shifts in these wavelengths due to the Doppler effect. The observed shift tells them the extent to which the light has been redshifted or blueshifted, thereby allowing them to calculate the speed of the star along the line of sight relative to Earth."),
    ("Provide a brief explanation of how black holes are formed and mention their key properties.", 
     "Black holes are formed when massive stars, typically those with a mass more than 20 times that of our sun, reach the end of their life cycle. After exhausting their nuclear fuel, these stars undergo a supernova explosion, which is the outer layers of the star being expelled into space. If the remaining core of the star is sufficiently massive, it will collapse under its own gravity, leading to the formation of a black hole. Key properties of black holes include their extremely strong gravitational pull, which is so strong that not even light can escape, an event horizon, which is the boundary surrounding black holes from where nothing can escape, and their singularity, the point at the center of the black hole where matter is crushed into infinite density."),
    ("Statement 1 | The number of elements in a finite group is always equal to the order of the group. Statement 2 | If G is a group, and H is a subgroup of G, then the order of H divides the order of G (Lagrange's Theorem). Answer Choices: (A) True, True (B) False, False (C) True, False (D) False, True", 
     "Statement 1: The number of elements in a finite group is always equal to the order of the group. This statement is true. By definition, the order of a group is the number of elements in the group. So, in a finite group, the number of elements is indeed equal to the order of the group. Statement 2: If G is a group, and H is a subgroup of G, then the order of H divides the order of G (Lagrange's Theorem). This statement is also true. Lagrange's Theorem states that if G is a finite group and H is a subgroup of G, then the order of H (the number of elements in H) divides the order of G (the number of elements in G). This is a fundamental result in group theory. The answer is A."),
    ("How is artificial gravity created through the rotation of a satellite or space station, and what acceleration is required for it to mimic Earth's gravity?", 
     """Artificial gravity in a rotating satellite or space station is achieved by the centripetal force that acts on the occupants toward the center of rotation. As the station rotates, friction with the floor accelerates the occupants tangentially. This tangential motion then causes the occupants to be pushed against the walls of the station, where the walls provide a normal force directed inward. This normal force simulates gravity and gives the sensation of being pulled toward the \"floor\" of the station.

To approximate Earth's gravity (9.81 m/s² or g), the satellite must be designed to spin at a rate that generates a centripetal acceleration of g at the rim where the occupants are located. It is important to note that this artificial gravity only exists relative to the rotating frame of reference; at the center of the station, there would be no such force, and occupants would experience weightlessness similar to that in free space. If one were to stop moving tangentially or reach the center, they would no longer be subject to the artificial gravity effect and would effectively be weightless."""),
    ("How does the concept of artificial gravity work according to scientific theories, and what methods have been used to simulate it?", 
     "Artificial gravity, as we understand it, is not a distinct invention but rather a byproduct of specific conditions and mechanisms. It is rooted in Albert Einstein's general theory of relativity, which posits that gravity is equivalent to acceleration. This means that when a person is in a frame of reference, like a car, experiencing constant acceleration, the force they feel can be likened to gravity.\n\nTo simulate the absence of gravity, or \"zero gravity,\" astronauts often undergo a training method called parabolic flights. In these flights, a plane ascends to a high altitude before entering a freefall, creating a short-lived weightless environment for passengers, typically lasting about 30 seconds at a time.\n\nFor simulating gravity in space, one theoretical approach involves using rotation. A large spaceship or space station could be designed with a rotating cylinder. As someone stands inside the cylinder, it would rotate at a specific speed, generating a centripetal force that pushes them against the inner wall. This force, acting in a circular direction, would create a sensation similar to gravity, pulling the individual \"downward\" along the wall of the cylinder. However, such a technology has not yet been implemented in a practical setting."),
    ("Describe how an airplane's lift force is generated according to the Bernoulli's principle.",
      "According to Bernoulli's principle, a fluid's pressure decreases as its speed increases, and vice versa. In the context of an airplane, this principle plays a significant role in generating lift force. The airplane wing, or airfoil, is designed in such a way that the air flowing over the top surface travels faster than the air flowing beneath the wing. This is due to the shape of the airfoil which forces the air on top to travel a longer distance in the same amount of time compared to the air below. As a result of the increased airspeed over the top surface, the pressure above the wing becomes lower than the pressure below the wing. This pressure differential creates an upward force on the wing known as lift, which counteracts the airplane's weight and enables it to become airborne.")
]
   

When looking at the results, the finetuned model often generated responses more similar to the groud truth than the original model. 
# Improving model performance.

## Model-centric approach.
There are a few sections in the notebook that perform hyperparameter optimization in order to attempt to find the best hyperparameters. However, limitations in available compute, as well as the large dimensinoality limits the number of hyperparameter configurations that can be tested. Given more time, several more configurations could be tested when performing successive halving, which could lead to better results.

Attempting to finetune larger model could also yield a better performance, but more parameters slows down the (already quite slow) inference and training.

## Data-centric approach
What data sources give a better performance depends on what the trained model is used for. Here the model is finetuned on the FineTome100K dataset that contains questions, and answers to thoose questions. In order to get better result it may be beneficial to find additional Question answering datasets. Some examples of such datasets are:

- SQuAD(stanford question answering dataset) containing questions about various wikipedia articles.
- Googles natural questions dataset.
- TriviaQA that contains several trivia Q&A pairs.

Another approach would be to incorporate theese datasets in the evaluation process, by for instance evaluating on questions from a similar dataset that the model wasn't finetuned on.

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



