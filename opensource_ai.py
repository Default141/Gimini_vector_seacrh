from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



# login()

def openthai():
    # Ensure CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Init Model
    model_path="openthaigpt/openthaigpt-1.0.0-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)

    # Prompt
    prompt = "วันนี้วันพุธ"
    llama_prompt = f"<s>[INST] <<SYS>>\nYou are Astrologer that will give daily forecast you will give forecast in thai<</SYS>>\n\n{prompt} [/INST]"
    inputs = tokenizer.encode(llama_prompt, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def langchain_demo():
    llm = HuggingFaceEndpoint(
        repo_id="openthaigpt/tinyllama_1b_th_en-pretrained-90B-tokens",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    chat = ChatHuggingFace(llm=llm, verbose=True)

    messages = [
    ("system", "You are a helpful translator. Translate the user sentence to Thai."),
    ("human", "I love programming."),
    ]

    result = chat.invoke(messages)

    # answer = result.get("answer", "No answer found.")
    # except Exception as e:
    #     print(f"Error during invocation: {e}")
    #     answer = "An error occurred."

    # Debug result
    print(f"Result: {result}")

openthai()
