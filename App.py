from setuptools import distutils
from InstructorEmbedding import  SentenceTransformer , torch 
import streamlit as st

def compara_frases(sentence_1 ,  sentence_2): 
    #------------------------------------defino modelo    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    #------------------------------------transforma en embeddings
    embeddings_1 = model.encode([sentence_1]) 
    embeddings_2 = model.encode([sentence_2])       
    
    #------------------------------------De embeddings a tensores torch    
    tensor_1 = torch.tensor(embeddings_1)
    tensor_2 = torch.tensor(embeddings_2)    

    #------------------------------------calcula y retorna similitud cos
    similitud_coseno = torch.nn.functional.cosine_similarity(tensor_1 , tensor_2 , dim=1)
    return similitud_coseno

def main():   
    #------------------------------------Streamlit 
    st.set_page_config(page_title="Similitud coseno" , page_icon=":books:")
    
    #------------------------------------palabras a comparar    
    sentence_1 = st.text_input("Frase 1")
    sentence_2 = st.text_input("Frase 2")

    if st.button("Procesar") :
        with st.spinner ("Procesando"):
            if sentence_1 and sentence_2:
                similitud_coseno = compara_frases(sentence_1 ,  sentence_2)                    
                st.write(f'Similitud del coseno "{sentence_1}" \n "{sentence_2}": {similitud_coseno.item()}')                 
            else:
                st.write("Por favor, ingrese ambas frases.")
    
if __name__ == '__main__':    
    main()