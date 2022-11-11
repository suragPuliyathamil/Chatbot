import torch
import os
import pickle
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
import datetime
# from utils import ChatBot
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'microsoft/DialoGPT-large'

# user_message = 'WHere are you now??'
#
# user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
#
# bot_output_ids = model.generate(user_input_ids, pad_token_id=tokenizer.eos_token_id)
#
# start_of_bot_message = user_input_ids.shape[-1]
# bot_message = tokenizer.decode(bot_output_ids[:, start_of_bot_message:][0], skip_special_tokens=True)
#
# print(bot_message)

class ChatBot:
    def __init__(self, model_name='microsoft/DialoGPT-large'):
        self.model, self.tokenizer = self.load_model(model_name)
        self.chat_history = []
        self.chat_history_ids = None

    def load_model(self, model_name):
        try:
            model = torch.load(r".\model.pt")
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            torch.save(model, r".\model.pt")
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer

    def get_reply(self, user_message):
    # save message from the user
        self.chat_history.append({
                'text':user_message,
                'time':str(datetime.datetime.now().time().replace(microsecond=0))
                })

    # encode the new user message to be used by our model
        message_ids = self.tokenizer.encode(user_message + self.tokenizer.eos_token, return_tensors='pt')

    # append the encoded message to the past history so the model is aware of past context
        if self.chat_history_ids is not None:
            message_ids = torch.cat([self.chat_history_ids, message_ids], dim=-1)

    # generated a response by the bot
        self.chat_history_ids = self.model.generate(
                message_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                  max_length=1000,
                  top_k=100,
                  top_p=0.95,
                  temperature=0.8,
                )


        decoded_message = self.tokenizer.decode(
                self.chat_history_ids[:, message_ids.shape[-1]:][0],
                skip_special_tokens=True
                )

    # save reply from the bot
        self.chat_history.append({
                'text':decoded_message,
                'time':str(datetime.datetime.now().time().replace(microsecond=0))
                })

        return decoded_message

app = FastAPI()

class Item(BaseModel):
    message: str

@app.post("/")
async def conversation(item:Item):

  # gets a response of the AI bot
  bot_reply = chatbot.get_reply(item.message)

  # returns the final HTML
  return {'message':bot_reply}

# initialises the chatbot model and starts the uvicorn app
if __name__ == "__main__":
  chatbot = ChatBot()
  uvicorn.run(app, host="0.0.0.0", port=8000)
