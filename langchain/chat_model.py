from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


from langchain.schema import HumanMessage


chat_model = Ollama(
    model="mistral:7b-instruct-q3_K_M",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
messages = [HumanMessage(content="Tell me about the history of AI")]
chat_model.predict("Tell me about the history of AI")


