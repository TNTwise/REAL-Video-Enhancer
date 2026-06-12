from pydantic import BaseModel


class VideoInfoClientInput(BaseModel):
    input_file: str
    output_file: str
