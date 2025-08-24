from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("7.DocumentLoader/airlines_flights_data.csv", encoding="utf-8")
data = loader.load()
print(data[0])
