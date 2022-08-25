import random
import re
from abc import ABCMeta
from difflib import SequenceMatcher
from typing import List

SOS = "<SOS>"
EOS = "<EOS>"


class TransformOp(metaclass=ABCMeta):
    def transform(self, str_value):
        pass

    def validate_target(self, str_value):
        pass


class InsertRandom():
    def __init__(self, inserting_str, prev_char):
        self.inserting_str = inserting_str
        self.prev_char = prev_char

    def transform(self, str_value):
        if self.prev_char == SOS:
            return self.inserting_str + str_value
        if self.prev_char == EOS:
            return str_value + self.inserting_str
        match = random.choice(list(re.finditer(re.escape(self.prev_char), str_value)))
        return str_value[: match.start()] + self.inserting_str + str_value[match.start() :]

    def validate(self, str_value):
        if self.prev_char in [SOS, EOS]:
            return True
        return len(re.findall(re.escape(self.prev_char), str_value)) > 0

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InsertRandom):
            return False
        return (
            self.inserting_str == o.inserting_str
            and self.prev_char == o.prev_char
        )

    def __str__(self) -> str:
        return f"InsertRandom({self.inserting_str}, {self.prev_char})"

    def __hash__(self) -> int:
        return hash(str(self))


class DeleteAll(TransformOp):
    def __init__(self, deleting_str):
        self.deleting_str = deleting_str

    def transform(self, str_value):
        return re.sub(re.escape(self.deleting_str), "", str_value)

    def validate(self, str_value):
        return self.deleting_str in str_value

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DeleteAll):
            return False
        return self.deleting_str == o.deleting_str

    def __str__(self) -> str:
        return f"DeleteAll({self.deleting_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class DeleteRandom(TransformOp):
    def __init__(self, deleting_str):
        self.deleting_str = deleting_str

    def transform(self, str_value):
        match = random.choice(list(re.finditer(re.escape(self.deleting_str), str_value)))
        return str_value[: match.start()] + str_value[match.end() :]

    def validate(self, str_value):
        return str_value.find(self.deleting_str) != -1

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DeleteRandom):
            return False
        return self.deleting_str == o.deleting_str 

    def __str__(self) -> str:
        return f"DeleteAt({self.deleting_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class DeleteAll(TransformOp):
    def __init__(self, deleting_str):
        self.deleting_str = deleting_str

    def transform(self, str_value):
        return re.sub(re.escape(self.deleting_str), "", str_value)

    def validate(self, str_value):
        return self.deleting_str in str_value

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DeleteAll):
            return False
        return self.deleting_str == o.deleting_str

    def __str__(self) -> str:
        return f"DeleteAll({self.deleting_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class ReplaceRandom(TransformOp):
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str
        
    def validate(self, str_value):
        return str_value.find(self.before_str) != -1

    def transform(self, str_value):
        match = random.choice(list(re.finditer(re.escape(self.before_str), str_value)))
        return str_value[: match.start()] + self.after_str + str_value[match.end() :]

    def validate_target(self, str_value):
        if self.after_str in str_value:
            index = str_value.find(self.after_str)
            if index != -1:
                return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ReplaceRandom):
            return False
        return (
            self.before_str == o.before_str
            and self.after_str == o.after_str
        )

    def __str__(self) -> str:
        return f"ReplaceRandom({self.before_str}, {self.after_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class ReplaceAll(TransformOp):
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        return re.sub(re.escape(self.before_str), self.after_str, str_value)

    def validate(self, str_value):
        return self.before_str in str_value

    def validate_target(self, str_value):
        return self.after_str in str_value and self.before_str not in str_value

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ReplaceAll):
            return False
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __str__(self) -> str:
        return f"ReplaceAll({self.before_str}, {self.after_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class TransformationLearner:
    def __init__(self) -> None:
        self.transform_ops = set()

    def fit(self, cleaned_values: List[str], error_values: List[str]):
        for cleaned_value, error_value in zip(cleaned_values, error_values):
            s = SequenceMatcher(None, cleaned_value, error_value)
            transforms = s.get_opcodes()

            for (tag, i1, i2, j1, j2) in transforms:
                if tag == "insert":
                    if i1 == 0:
                        self.transform_ops.add(InsertRandom(error_value[j1:j2], SOS))
                    elif i2 == len(cleaned_value):
                        self.transform_ops.add(InsertRandom(error_value[j1:j2], EOS))
                    else:
                        self.transform_ops.add(InsertRandom(error_value[j1:j2], cleaned_value[i1 - 1]))
                elif tag == "delete":
                    self.transform_ops.add(
                        DeleteRandom(
                            cleaned_value[i1:i2]
                        )
                    )
                    self.transform_ops.add(
                        DeleteAll(cleaned_value[i1:i2])
                    )
                elif tag == "replace":
                    self.transform_ops.add(
                        ReplaceRandom(
                            cleaned_value[i1:i2],
                            error_value[j1:j2],
                        )
                    )

                    self.transform_ops.add(ReplaceAll(cleaned_value[i1:i2], error_value[j1:j2]))

        return list(self.transform_ops)