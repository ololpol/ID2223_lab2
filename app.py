import gradio as gr
from huggingface_hub import InferenceClient
import random
from llama_cpp import Llama


# -------------------------------------------------------
# Configure Hugging Face models
# -------------------------------------------------------

MODEL_1 = "oloflil/model"
MODEL_2 = "meta-llama/Llama-3.2-1B-Instruct"


#client1 = InferenceClient(model=MODEL_1, token=HF_TOKEN)
client1 = None
client2 = None

c1_load = False
c2_load = False


# ---------------------------------------
# Predefined messages for random selection
# ---------------------------------------
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
    
def query_model(client, prompt, use_ggpg = False):
    if use_ggpg:
        result = client.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return result["choices"][0]["message"]["content"]
    else: 
        result = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return result.choices[0].message["content"]



def chat(prompt, hf_token: gr.OAuthToken):
    """Return responses from two different LLMs."""
    if c1_load == False:
        client1 = Llama.from_pretrained(
            repo_id=MODEL_1,
            filename="unsloth_lora_modelmodel-3b-Q8_0.gguf",
            token=hf_token.token
        )
    if c2_load == False:
        client2 = InferenceClient(model=MODEL_2, token=hf_token.token)
    response1 = query_model(client1, prompt, True)
    response2 = query_model(client2, prompt)

    return response1, response2


def send_random_message(hf_token: gr.OAuthToken):
    """Pick a random question, show ground truth, and query both LLMs."""
    question, truth = random.choice(QA_PAIRS)
    if c1_load == False:
        client1 = Llama.from_pretrained(
            repo_id=MODEL_1,
            filename="unsloth_lora_modelmodel-3b-Q8_0.gguf",
            token=hf_token.token
        )
    if c2_load == False:
        client2 = InferenceClient(model=MODEL_2, token=hf_token.token)
    r1 = query_model(client1, question, True)
    r2 = query_model(client2, question)

    return question, truth, r1, r2


# -------------------------------------------------------
# Gradio UI
# -------------------------------------------------------
with gr.Blocks() as demo:
    c1_load = False
    c2_load = False

    gr.Markdown("# 🤖 Multi-LLM Chat with Ground-Truth from FineTome")

    # User message box
    gr.LoginButton()
    user_input = gr.Textbox(label="Your message")

    # Shared output fields
    out1 = gr.Textbox(label=f"Response from {MODEL_1}")
    out2 = gr.Textbox(label=f"Response from {MODEL_2}")

    # User send button
    gr.Button("Send").click(chat, inputs=user_input, outputs=[out1, out2])

    # Random QA section
    gr.Markdown("## 🎲 Random Question Benchmark")

    rand_btn = gr.Button("Ask Random Question")

    rand_question = gr.Textbox(label="Random Question", interactive=False)
    rand_truth = gr.Textbox(label="Ground Truth Answer", interactive=False)

    # Random question fills the SAME two LLM output boxes
    rand_btn.click(
        send_random_message,
        inputs=None,
        outputs=[rand_question, rand_truth, out1, out2],
    )

demo.launch()