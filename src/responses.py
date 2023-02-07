import json
import random
import pandas as pd
from numpy.linalg import norm
import numpy as np
import openai
import re
import time
import datetime

def get_config() -> dict:
    import os
    # get config.json path
    config_dir = os.path.abspath(__file__ + "/../../")
    config_name = 'config.json'
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
        
def save_chat(author, message, timestamp, vector):
    meta = {'timestamp': timestamp, 'author': author, 'message': message }
    new_chat = {'metadata': meta, 'id': db.iloc[-1]['id']+1, 'embedding': vector }
    # save to db
    db.loc[len(db.index)] = new_chat
    # append to file
    with open(chat_json_file, 'w') as f:
        db.to_json(f, orient='records')

        
def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity

def similarity_score(row, vector):
    if row["embedding"] == vector:
        return float('-inf')
    return similarity(row["embedding"], vector)

def process_entry(index, str_filter):
    vector = db.loc[index, "embedding"]
    for x in range(1,10):
        if index + x in db.index and db.loc[index + x, "metadata"]["author"] == str_filter:
            score = similarity(vector, db.loc[index + x, "embedding"])
            if score > mem_temp:
                response = db.loc[index + x, "metadata"]
                message = db.loc[index, "metadata"]
                return message, response
    return None

async def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

async def gpt3_completion(prompt, user, engine='text-davinci-003', temp=0.6, tokens=600):
    max_retry = 5
    retry = 0
    stops = [bot_name+':',user+':']
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    print(prompt)
    while True:
        try:
            response = openai.Completion.create(engine=engine, prompt=prompt, temperature=temp, max_tokens=tokens, stop=stops)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            time.sleep(1)

def fetch_memories(vector, max_count):
    #only get chats that are responded to by filter
    scores = db.copy()
    scores["score"] = scores.apply(similarity_score, axis=1, vector=vector)
    scores = scores.sort_values(by='score', ascending=False)
    chat_set = [entry for entry in scores.iloc[:100].apply(lambda x: process_entry(x.name, str_filter), axis=1) if entry is not None]
    return chat_set[:max_count]

def fetch_valid_lore(vector, count):
    scores = pd.read_json(facts_file)
    scores["score"] = scores.apply(similarity_score, axis=1, vector=vector)
    scores = scores.sort_values(by='score', ascending=False)
    print(scores)
    lore = scores.head(count)['message'].apply(lambda x: x['lore_item']).tolist()
    return lore


# CONFIG DATA
config = get_config()
openai.organization = config["openAI_org"]
openai.api_key = config["openAI_key"]
bot_name = config["bot_name"]
str_filter = config["str_filter"]
mem_temp = config["mem_temp"] #change for more/less memories, may increase token usage
# LOCAL DATA
chat_json_file = 'data/DiscordEmbeddings.json'
facts_file = 'data/darcy.json'
chat_prompt_file = 'prompts/ChatPrompt.txt'
db = pd.read_json(chat_json_file)

async def handle_response(user, message) -> str:
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    vector = await gpt3_embedding(message)
    save_chat(user, message, timestamp, vector)
    
    #search for memories
    memories = fetch_memories(vector, 3)  # pull relevant Darcy responses
    
    mem_block =''
    for m in memories:
        mem_block += '\n%s: %s' % (m[0]['author'],m[0]['message']) + '\n%s: %s\n' % (bot_name,m[1]['message'])
        
    #get random facts
    facts = fetch_valid_lore(vector, 5)
    facts_block =''
    for f in facts:
        facts_block += '\n%s' % f
    
    recent_convo = db[-5:]["metadata"].tolist() #get last 5 messages
    convo_block = ''
    for c in recent_convo:
        convo_block += '\n%s: %s' % (c["author"],c["message"])
    
    prompt = open_file(chat_prompt_file).replace('<<NOTES>>', mem_block).replace('<<FACTS>>', facts_block).replace('<<CONVERSATION>>', convo_block)
    output = await gpt3_completion(prompt, user)
    #save message
    vector = await gpt3_embedding(output)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    save_chat(bot_name, output, timestamp, vector)
    
    return output.replace(":darcy:","<:darcy:613735309343064085>").replace('!','.')

async def handle_lore(message) -> str:

    with open(facts_file, 'r') as f:
        facts_data = json.load(f)
    vector = await gpt3_embedding(message)
    data = {"message": {"lore_item": message}, "embedding": vector}
    facts_data.append(data)
    with open(facts_file, 'w') as f:
        json.dump(facts_data, f)
    output = "Noted. Thanks, bud."
    return output