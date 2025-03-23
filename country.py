from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from data_self_generate import data_generation, data_generation2
from out_put_parser import parser
from prompt_repo import prompt_country, prompt_bidvalue, prompt_gender
from langchain_core.runnables import RunnableParallel

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
### bid value
bidvalue_chains =template_promt_generation(promt= prompt_bidvalue)
### gender
gender_chains =template_promt_generation(promt= prompt_gender)

## combine_chain
combine_chains = RunnableParallel(country = country_chains, bid_value = bidvalue_chains, gender = gender_chains)

result = combine_chains.invoke({"data" : data,
                    "your_column_input" : "country",
                    ## for bid
                    "main_col" : "Bid_Raw",
                    "sub_colmn" : "bid value",
                    ## for gender
                     "yr_col" : "Raw_Data",
                    "sub_item" : "gender"
                    })

print(result['country']['result'])
print('\n\n')
print(result['bid_value']['result'])
print('\n\n')
print(result['gender']['result'])
