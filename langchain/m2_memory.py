from langchain.llms import CTransformers
from loguru import logger
from langchain.prompts import ChatPromptTemplate

from langchain.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# Load the Model
config = {"temperature":0.0,"gpu_layers":50}
model_path = "models/zephyr-7b-alpha.Q4_K_M.gguf"

try:
    logger.info("loading the model...")
    llm = CTransformers(model=model_path,config=config)
except Exception as e:
    logger.info(f"Exception Loading the model : {e}")




memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

while True:
    input_str = str(input("Enter Your Question : "))
    logger.info(conversation.predict(input=input_str))