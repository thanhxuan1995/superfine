from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from data_self_generate import data_generation, data_generation2
from out_put_parser import parser
from prompt_repo import prompt_country, prompt_bidvalue, prompt_gender

parser_out = parser()
llm_model = ChatGroq(model="gemma2-9b-it")
data = data_generation2()
def template_promt_generation(promt):
    prompt = promt
    templates = PromptTemplate(template=prompt, 
                                         partial_variables={'format_instruction' : parser_out.get_format_instructions()},
                                         input_variables=['data'])
    chain = templates | llm_model | parser_out
    return chain

### country
country_chains = template_promt_generation(promt= prompt_country)
print(country_chains.invoke({"data" : data,
                    "your_column_input" : "country",
                    }))
### bid value
bodvalue_chains =template_promt_generation(promt= prompt_bidvalue)
print(bodvalue_chains.invoke({"data" : data,
                    "your_column_input" : "Bid_Raw",
                    "sub_colmn" : "bid value"}))

gender_chains =template_promt_generation(promt= prompt_gender)
print(gender_chains.invoke({"data" : data,
                    "your_column_input" : "Raw_Data",
                    "sub_colmn" : "gender"}))