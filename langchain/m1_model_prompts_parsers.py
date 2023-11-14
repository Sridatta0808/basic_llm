from langchain.llms import CTransformers
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



# Load the Model
config = {"temperature":0.0,"gpu_layers":100}
model_path = "models/zephyr-7b-alpha.Q4_K_M.gguf"

try:
    llm = CTransformers(model=model_path,config=config)
except Exception as e:
    logger.info(f"Exception Loading the model : {e}")


def get_completion(prompt):
    messages = f"""role":"user", "content": {prompt}"""
    response = llm(messages)
    logger.info(f"Response:{response}")

"""
Section 1 : Test the Model with Random Questions.
"""
Flag = True
logger.info(f"Type 'quit' to Stop")
while Flag:
    logger.info(f"Enter your Question:")
    question = str(input())
    if question == "quit":
        break
    else:
        get_completion(question)


"""
Section 2 : Give an Example of a Customer Email and then ask the model to change the Tone of the mail.
"""
# Example  1
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# Set the Style
style = """American English \
in a calm and respectful tone
"""

prompt = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{customer_email}``` \
"""

# Run the Prompt
#get_completion(prompt)

"""
Section 3 : Output Parsers, Parse in the desired Format.
"""
# Create a Prompt Tempalte
prompt_tempalte = ChatPromptTemplate.from_template(prompt)
# logger.info(f"Chat Template Object : {prompt_tempalte}")
# logger.info(f"PROMPT TEMPALTE : {prompt_tempalte.messages[0].prompt}")

# Custom Messages to Generate the prompt
customer_messages  =prompt_tempalte.format_messages(
    style=style,
    customer_email = customer_email
)

logger.info(customer_messages)




# Example 2 
service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""


# prompt = """Translate the text \
# that is delimited by triple backticks \
# into a style that is {service_style_pirate}. \
# text: ```{service_reply}``` \
# """
# prompt_template = ChatPromptTemplate.from_template(prompt)

# service_messages = prompt_template.format_messages(
#     service_style_pirate=service_style_pirate,
#     service_reply=service_reply)

# print(service_messages[0].content)

# res = llm(service_messages[0].content)
# print(res)

# Example 3 
# {
#   "gift": False,
#   "delivery_days": 5,
#   "price_value": "pretty affordable!"
# }

# customer_review = """\
# This leaf blower is pretty amazing.  It has four settings:\
# candle blower, gentle breeze, windy city, and tornado. \
# It arrived in two days, just in time for my wife's \
# anniversary present. \
# I think my wife liked it so much she was speechless. \
# So far I've been the only one using it, and I've been \
# using it every other morning to clear the leaves on our lawn. \
# It's slightly more expensive than the other leaf blowers \
# out there, but I think it's worth it for the extra features.
# """

# review_template = """\
# For the following text, extract the following information:

# gift: Was the item purchased as a gift for someone else? \
# Answer True if yes, False if not or unknown.

# delivery_days: How many days did it take for the product \
# to arrive? If this information is not found, output -1.

# price_value: Extract any sentences about the value or price,\
# and output them as a comma separated Python list.

# Format the output as JSON with the following keys:
# gift
# delivery_days
# price_value

# text: {text}
# """

# prompt_template = ChatPromptTemplate.from_template(review_template)
# messages = prompt_template.format_messages(text=customer_review)
# print(messages)
# res = llm(messages[0].content)
# print(f"OUTPUT : {res} ")


# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser



# gift_schema = ResponseSchema(name="gift",
#                              description="Was the item purchased\
#                              as a gift for someone else? \
#                              Answer True if yes,\
#                              False if not or unknown.")
# delivery_days_schema = ResponseSchema(name="delivery_days",
#                                       description="How many days\
#                                       did it take for the product\
#                                       to arrive? If this \
#                                       information is not found,\
#                                       output -1.")
# price_value_schema = ResponseSchema(name="price_value",
#                                     description="Extract any\
#                                     sentences about the value or \
#                                     price, and output them as a \
#                                     comma separated Python list.")

# response_schemas = [gift_schema, 
#                     delivery_days_schema,
#                     price_value_schema]

# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# format_instructions = output_parser.get_format_instructions()

# print(format_instructions)


# review_template_2 = """\
# For the following text, extract the following information:

# gift: Was the item purchased as a gift for someone else? \
# Answer True if yes, False if not or unknown.

# delivery_days: How many days did it take for the product\
# to arrive? If this information is not found, output -1.

# price_value: Extract any sentences about the value or price,\
# and output them as a comma separated Python list.

# text: {text}

# {format_instructions}
# """

# prompt = ChatPromptTemplate.from_template(template=review_template_2)

# messages = prompt.format_messages(text=customer_review, 
#                                 format_instructions=format_instructions)

# res = llm(messages[0].content)

# output_dict = output_parser.parse(res)
# print(type(output_dict))