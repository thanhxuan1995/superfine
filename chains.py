from langchain_groq import ChatGroq
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
import pandas as pd
from langchain.agents import AgentType, AgentExecutor, initialize_agent
from langchain_core.tools import Tool
from typing import List, Dict
from io import StringIO
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, RootModel


class OurputParser(RootModel[Dict[str, str]]):
    pass


class LayerFinal:
    def __init__(self, file_path: str, col_update: Dict):
        self.file_path = file_path
        self.col_update = col_update

    def read_csv(self):
        df = pd.read_csv(self.file_path)
        return df

    @staticmethod
    def LLMModel(input_data: str) -> Dict:
        system_prompt = SystemMessagePromptTemplate.from_template(
            "System : You are an expert in semantic text segmentation"
        )
        ai_prompt = AIMessagePromptTemplate.from_template(
            "AI: Given a list of short phrases, group them by semantic similarity."
            "Each group must use one of the original phrases as the group label (i.e., the representative of that group)"
            "Output a dictionary where each key is a phrase from the list, and its value is the representative phrase for its group"
            "Do not add any explanation or extra text.\n"
            "{format_instructions}"
        )
        human_prompt = HumanMessagePromptTemplate.from_template(
            "Human: here is the input list {input_data}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_prompt, ai_prompt, human_prompt]
        )
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.1)
        parser = JsonOutputParser(pydantic_object=OurputParser)
        parser_format = parser.get_format_instructions()
        chain = prompt | llm | parser
        res = chain.invoke(
            {
                "input_data": input_data,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return res

    def callToUpdate(self):
        df = self.read_csv()
        for col in self.col_update:
            input_list = [x.strip() for x in df[col].astype(str).to_list()]
            categories = LayerFinal.LLMModel(input_data=input_list)
            df[col] = df[col].apply(lambda x: categories[x.strip()])
        return df


if __name__ == "__main__":
    file_path = r"C:\Users\a\Downloads\superfine\langchain\standard_categories.csv"
    col_update = ["art_style", "color_theme"]
    df = LayerFinal(file_path=file_path, col_update=col_update).callToUpdate()
    df.to_csv("Xuan_print_categories.csv", index=False)
