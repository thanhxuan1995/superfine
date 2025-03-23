from langchain_core.output_parsers import JsonOutputParser
from typing import  Optional, List, Any
from pydantic import BaseModel, Field

class objectformat(BaseModel):
    """result print to user"""
    result : List[Any] = Field(description=" The result as a list")

def parser():
    parser_result = JsonOutputParser(pydantic_object= objectformat)
    #print(parser_result.get_format_instructions())
    return parser_result

if __name__ == '__main__':
    print(parser())