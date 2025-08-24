from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    lastname: Optional[str] = "Kumar"
    email: EmailStr # Validates the Error types
    cgpa: float = Field(gt=0,lt=10, default=5, description= "A decimal value representing the cgpa od the student")
    # Can also add reg exp here - Explore later


newStudent = {"name":"Mohan","age": 32, "email": "abc@gmail.com", "cgpa":9}

student = Student(**newStudent)
print(student)

# Converting the output to Dict, JSON
student_dict = student.model_dump()

student_JSON = student.model_dump_json()

print(student_dict)
print(student_JSON)