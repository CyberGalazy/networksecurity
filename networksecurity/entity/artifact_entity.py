from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str