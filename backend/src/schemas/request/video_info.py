from pydantic import BaseModel


class InputVideoInfoClientInput(BaseModel):
    input_file: str


class OutputVideoInfoClientInput(BaseModel):
    output_file: str
