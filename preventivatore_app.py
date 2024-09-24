import streamlit as st
from io import StringIO
import pandas as pd
from openai import OpenAI
import os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
import json
import re
import ast
import openai
from pydantic import BaseModel, Field, constr
from typing import Optional
# Configura l'API di OpenAI

# Carica la tabella dei prodotti
df = pd.read_csv('catalogo_con_link.csv', sep = ';')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def extract_parameters(user_input):
    prompt = f"""
                Estrai i seguenti parametri dalla richiesta dell'utente, considerando solo informazioni pertinenti all'arredamento di una casa. Non includere informazioni non correlate o termini generici.

                Parametri da estrarre:
                {{
                  "nome": "opzionale, il nome specifico dell'articolo di arredamento (es. elegance, comfort, classic, modern). Non considerare nomi generici o comuni come 'mobile', 'articolo', 'prodotto', o 'arredamento'.",
                  "tipo": "opzionale, il nome generico del tipo di arredamento, come ad esempio 'divano', 'letto', 'sedia', 'armadio', 'scrivania', 'tavolo', 'comodino'. I termini 'mobile', 'arredamento', 'articolo', 'prodotto' non fanno parte di questo parametro, ignorali.",
                  "zona": "opzionale, l'area della casa in cui l'articolo sarà posizionato (es. soggiorno, cucina, camera da letto).",
                  "materiale": "opzionale, il materiale dell'articolo di arredamento (es. legno, metallo, tessuto).",
                  "misure": "opzionale, le misure dell'articolo (es. larghezza, altezza, profondità).",
                  "colore": "opzionale, il colore dell'articolo.",
                  "prezzo": "opzionale, il range di prezzo dell'articolo.",
                  "link": "opzionale, true se serve un link per maggiori informazioni o acquisto, false altrimenti."
                }}

                Se un parametro non è specificato, restituisci None in corrispondenza di quel parametro. Non inventare valori.

                Restituisci solo il dizionario dei parametri in formato JSON. Non aggiungere spiegazioni, codice, o altre informazioni.

                Esempio di risposta:
                {{
                  "nome": None,
                  "tipo": None,
                  "zona": None,
                  "materiale": None,
                  "misure": None,
                  "colore": None,
                  "prezzo": None,
                  "link": None
                }}

                Richiesta: {user_input}
                """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0
    )
    extracted_params = response.choices[0].text.strip()

    # Verifica se extracted_params è già un dizionario
    if isinstance(extracted_params, dict):
        valid_params = extracted_params
    else:
        # Verifica se extracted_params è una stringa e rimuovi eventuali spazi extra
        extracted_params = extracted_params.strip()
        extracted_params = '{' + extracted_params.split('{')[1]
        extracted_params = extracted_params.split('}')[0] + '}'
        #st.write(f"Extracted Params: {extracted_params}")

        # Tenta di valutare solo se sembra essere un dizionario o una lista
        if extracted_params.startswith("{") and extracted_params.endswith("}"):
            try:
                valid_params = ast.literal_eval(extracted_params)
            except (SyntaxError, ValueError) as e:
                st.error(f"Errore nella risposta dell'API: {e}")
                valid_params = {}
        else:
            st.error("La risposta non è un dizionario valido.")
            valid_params = {}

    return valid_params


def filter_with_llm(parameters, df):
    # Usa l'LLM per determinare i filtri da applicare alla tabella
    messages = [
        {"role": "system", "content": "Sei un assistente che filtra dati sulla base delle istruzioni date."},
        {"role": "user", "content": f"""
            Filtra la seguente tabella di prodotti utilizzando i parametri forniti. La corrispondenza deve essere semantica, quindi includi prodotti che usano sinonimi o termini equivalenti ai valori dei parametri specificati. Tuttavia, non devi inventare nuovi prodotti o alterare quelli esistenti. Usa solo le informazioni presenti nella tabella.

            Parametri di filtraggio:
            {parameters}

            Istruzioni di filtraggio:
            - "nome": Includi prodotti che contengono esattamente questo nome o sinonimi pertinenti, senza inventare nuovi nomi.
            - "zona": Includi prodotti destinati esattamente alla zona indicata o a zone sinonime (es. "soggiorno" può includere "living"), senza aggiungere nuove zone.
            - "materiale": Includi prodotti che corrispondono al materiale specificato o a materiali equivalenti, ma non creare materiali nuovi.
            - "misure": Includi prodotti che rispettano le misure specificate, mantenendo una corrispondenza stretta, senza approssimazioni o alterazioni delle dimensioni.
            - "colore": Includi prodotti che hanno esattamente il colore indicato o tonalità simili, senza aggiungere nuovi colori.
            - "prezzo": Includi prodotti che rientrano nel range di prezzo indicato, senza inventare nuovi prezzi.
            - "link": Se True, includi solo i prodotti che hanno un link disponibile; se False, ignora questo parametro.

            **Importante**: Non inventare prodotti, nomi, materiali o caratteristiche. Utilizza solo i dati esistenti nella tabella fornita.
            "Restituisci i risultati in formato tabellare con le seguenti colonne: Nome, Zona, Materiale, Misure, Colore, Prezzo, Link. Non aggiungere altro. "
             "Esempio di output richiesto:\n"
            | Nome | Tipo | Zona | Materiale | Misure | Colore | Prezzo | Link |
            | ---- | ---- | ---- | --------- | ------ | ------ | ------ | ---- |
            | Tavolo da cucina | Tavolo | Cucina | Legno | 120x60x75 cm | Marrone | €200 | [link](#) |

            Se non trovi prodotti non restituire nessuna tabella, ma dì solo che non hai trovato prodotti corrispondenti ai criteri di ricerca.
            Tabella:
            {df.to_string(index=False)}
        """}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        #max_tokens=500
    )

    # Converte la risposta in un dataframe filtrato (richiede parsing della risposta)

    filtered_data = response.choices[0].message.content.strip()
    # Esempio di conversione, va adattato al formato esatto dell'output
    #filtered_df = pd.read_csv(StringIO(filtered_data))

    return filtered_data

def inizializza_stato():
    if 'dizionario' not in st.session_state:
        st.session_state.dizionario = {
            'nome': None,
            'tipo': None,
            'zona': None,
            'materiale': None,
            'misure': None,
            'colore': None,
            'prezzo': None,
            'link': None
        }
    if 'messaggio_elaborato' not in st.session_state:
        st.session_state.messaggio_elaborato = False


    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'user_input' not in st.session_state:
        st.session_state.user_input = None


def reset_chat():
    st.session_state.dizionario = {
        'nome': None,
        'tipo': None,
        'zona': None,
        'materiale': None,
        'misure': None,
        'colore': None,
        'prezzo': None,
        'link': None
    }
    st.session_state.messages = []
    st.session_state.messaggio_elaborato = False

    if "initialized" in st.session_state:
        del st.session_state["initialized"]

    if "user_input" in st.session_state:
        del st.session_state["user_input"]
        st.session_state.user_input = None


def aggiorna_dizionario(valori):
    for chiave, valore in valori.items():
        if valore is not None:
            st.session_state.dizionario[chiave] = valore




def main():
    st.set_page_config(page_title="Assistente Catalogo", layout="wide")



    if "initialized" not in st.session_state:
        inizializza_stato()
        st.session_state.initialized = True  # Flag per evitare ripetizioni

    with st.sidebar:
        st.image('poltrona_logo.png', use_column_width=False, width=200)
        st.markdown('### Benvenuto nel tuo assistente virtuale *IntelliCasa*!')

        st.divider()
        st.write("Oppure usa uno di questi suggerimenti:")

        if st.button("Living"):
            user_input = "Quali sono gli articoli disponibili per la zona living?"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})
        if st.button("Cucina"):
            user_input = "Quali sono gli articoli disponibili per la cucina?"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})
        if st.button("Studio"):
            user_input = "Quali sono gli articoli disponibili per lo studio?"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})

        st.divider()
        # Bottone per azzerare tutto e ripartire

        if st.button("Nuova chat"):
            reset_chat()

    # Gestione dell'interazione dell'utente tramite chat
    st.header('Assistente Catalogo')
    st.divider()

    with st.chat_message("assistant", avatar = 'architetto_icon.png'):
        st.write("Ciao! Fammi una domanda sul catalogo." )

    # Input per la chat
    if user_input := st.chat_input('Fammi la tua domanda'):
        st.session_state.user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})

    if st.session_state.user_input:
        # estraggo la checklist
        temp_dict = extract_parameters(st.session_state.user_input)
        aggiorna_dizionario(temp_dict)
        #st.session_state.dizionario = extract_parameters(st.session_state.user_input)
        #st.session_state.messages.append({"role": "assistant", "content": st.session_state.dizionario, "avatar": "architetto_icon.png"})


        estratti = {k: v for k, v in st.session_state.dizionario.items() if v not in (None, '', False)}
        non_estratti = {k for k, v in st.session_state.dizionario.items() if v in (None, '', False) and k not in 'link'}
        if estratti.get("tipo", "").strip().lower() not in [
                                                            'armadio',
                                                            'comodino',
                                                            'divano',
                                                            'letto',
                                                            'libreria',
                                                            'poltrona',
                                                            'scrivania',
                                                            'sedia',
                                                            'sedia da ufficio',
                                                            'tavolo'
                                                        ]:
            st.session_state.dizionario['tipo'] = None
            estratti['tipo'] = None




        with st.spinner("Sto elaborando la tua richiesta..."):
            if not estratti:
                st.session_state.messages.append({"role": "assistant", "content": "Sembra che la tua ricerca non corrisponda a nessun prodotto nel catalogo. Puoi essere più preciso?", "avatar": 'icons8-architetto-donna-80.png'})
            else:
                # esegui la prima elaborazione sulla tabella
                filtered_products = filter_with_llm(st.session_state.dizionario, df)

                assistant_message = (
                    "Questi sono gli articoli che ho trovato nel catalogo:\n\n"
                    f"{filtered_products}\n\n"  # Aggiungi una linea extra per lo spazio
                )

                assistant_message += (
                    "\nSe vuoi **perfezionare la tua ricerca**, puoi darmi ulteriori dettagli su:\n\n"
                )

                # Converti i parametri a punti elenco
                bullet_points = "\n".join([f"- *{key}*" for key in non_estratti])

                #ESCAPE SU "TIPO" --> se l'estrattore dei parametri ha inserito il valore "mobile" nella chiave "tipo",
                #i bullet points della risposta riportano comunque la chiave. 
                if estratti.get("tipo") not in ['Armadio',
                                                 'Comodino',
                                                 'Divano',
                                                 'Letto',
                                                 'Libreria',
                                                 'Poltrona',
                                                 'Scrivania',
                                                 'Sedia',
                                                 'Sedia da Ufficio',
                                                 'Tavolo']:
                    bullet_points += "\n- *tipo*"


                assistant_message += bullet_points

                # Aggiungi il messaggio alla lista dei messaggi
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_message, "avatar": 'architetto_icon.png'}
                )
        st.session_state.messaggio_elaborato = True

        for message in st.session_state.messages:
            if message['role'] == 'assistant':
                with st.chat_message('assistant', avatar='architetto_icon.png'):
                    st.write(message["content"])
            else:
                with st.chat_message('user', avatar='user_icon.png'):
                    st.write(message["content"])

if __name__ == "__main__":
    main()
