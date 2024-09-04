## 1. Import langchain function
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import transformers
## 2. create function to invoke model
def titan_llm():
        llm=BedrockChat(
        model_id='amazon.titan-text-express-v1',
        model_kwargs={
        'temperature':0.5,
        'topP':0.9,
        'maxTokenCount':100
        }
        )
        return llm
#test the model
#        return llm.invoke(input_text)
#response = titan_llm.invoke("what is Langchain")
#print(response)


## 3. create memory function for chatbot
def create_memory():
        llm=titan_llm()
        memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100
        ) # this is to maintain a summary of previous memory
        return memory


## 4. create a chat client function to run the chatbot
def get_chat_response(input_text,memory): 
        llm=titan_llm()
        conversationalchain_with_memory=ConversationChain(  # create aconversation chain
        llm=llm,                                  # using AWS Bedrock  
        memory=memory,                            # using summarization memory 
        verbose=True        # printout the internat states of the running chain
        )

        chat_response=conversationalchain_with_memory.invoke(input=input_text)  #pass the user message and memory to the model
        return chat_response['response']