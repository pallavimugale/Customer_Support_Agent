import faiss # used for fast vectore search & cluster .search fast in huge dataset, image /audio/text embeding 
import numpy as np #lib in py for numerical computation

from typing import Tuple, List #tuple is for immutable storing var 
import pandas as pd #lib for data analysis and manupulation handle sql ,csv, excel,sql files
#from langchain.embeddings import OpenAIEmbeddings #ou want to generate vector embeddings from text using OpenAI’s models inside a LangChain pipeline.
from openai import OpenAI # used of ai mfeature

from sentence_transformers import SentenceTransformer #(Hugging Face) – Free

from groq import Groq
import config

#import config #contains openAI Key information : Open AI key is not using no need to add

#API_KEY=config.OPENAI_API_KEY # it is paid one so will try with other api key for free version

def create_embeddings(df: pd.DataFrame, column_name: str, model: str)->np.ndarray:

  # created the embedding using open ai key and model 
  #embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)

  #create embedding using sentence_transformers for free
  embedder= SentenceTransformer(model)

  #encode the text data in speciallied col using sentance tranformar model
  df[f"{column_name}_vector"] =df[column_name].apply(lambda x:embedder.encode(x))

  # stack the encoded vector in numpy array
  vectors= np.stack(df[f"{column_name}_vector"].values)

  return vectors

#fun to create faiss index , add provided vectors into index and save it in files
def create_index(vectors: np.ndarray, index_file_path: str)->faiss.Index:

  #get the dimention 
  dimension= vectors.shape[1]

  #create faiss index with l2 distance metrix (cosine similarity)
  index =faiss.IndexFlatL2(dimension)

  #add vector to index 
  index.add(vectors)

  #save index to the file
  faiss.write_index(index,index_file_path)

  print("faiss index is created and added to the file")
  return index

#cal similarity between query and set of indexed vectors
# Args:
#         query (str): The query string.
#         index (faiss.Index): The FAISS index used for searching.
#         model (str): The name of the OpenAI embedding model used to create embedding.
#         k (int, optional): The number of most similar vectors to retrieve. Defaults to 3.

    # Returns:
    #     tuple: A tuple containing two arrays - D and I.
    #         - D (numpy.ndarray): The distances between the query vector and the indexed vectors.
    #         - I (numpy.ndarray): The indices of the most similar vectors in the index.

def semantic_similarity(query: str, index: faiss.Index, model: str, k: int=3)->Tuple[np.ndarray, np.ndarray]:
  model=SentenceTransformer(model)

  #embed the query
  query_vector=model.encode(query)
  query_vector=np.array([query_vector]).astype('float32')

  #search the faiss index
  D, I=index.search(query_vector,k)

  return D, I


#call the llm model based on the query and responce list

def call_llm(query: str, responses: list[str])->str:

   #assuming your KEY is saved in your environment variable as described in the Readme
   #client=OpenAI(api_key=API_KEY)
   client = Groq(api_key=config.GROQ_API_KEY)

    #below is the promt templete that we have to give to llm for better ans and responce
   messages=[
     {"role":"system" , "content": "You are a helpful assistant and help answer customer's query and request. Answer on the basic of provided context only."},
     {"role":"user" , "content": f'''On the basic of  input customer query determine or suggest the following things about the input query:{query}:
                                  1. Urgency of the query based on the input query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
                                          2. Categorize the input query into sales, product, operations etc. Only tell me the category.
                                          3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
                                          If the input query form customer is not clear then ask a follow up question.  
      '''}
     ]
   response = client.chat.completions.create(model="llama3-8b-8192", messages=messages, temperature=0)
    
   return response.choices[0].message.content
   
 




   

