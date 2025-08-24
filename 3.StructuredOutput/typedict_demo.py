from typing import TypedDict

class person(TypedDict):
    name: str
    age: int

new_person : person = {"name": 46, "age": 32}
print(new_person["name"])

new_person : person = {"name": "Mohan", "age": 32}
print(new_person["name"])