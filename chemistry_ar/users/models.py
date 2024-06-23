import json
import numpy as np


class User:
    def __init__(self, name, face_encoding, level):
        self.name: str = name
        self.face_encoding = face_encoding
        self.level: int = level

    def to_dict(self):
        return {
            "name": self.name,
            "face_encoding": self.face_encoding,
            "level": self.level,
        }


class UserEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, User):
            return {
                "name": o.name,
                "face_encoding": o.face_encoding.tolist()
                if isinstance(o.face_encoding, np.ndarray)
                else o.face_encoding,
                "level": o.level,
            }
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
