from langchain_community.document_loaders import WebBaseLoader
url = "https://www.amazon.com/Apple-2024-MacBook-15-inch-Laptop/dp/B0DLHB9X2G/ref=sr_1_1_sspa?crid=2WKBIBESFZJ0W&dib=eyJ2IjoiMSJ9.wz5R65-5Xsqrl2EawEveni2j2NwKoii4mIyjS3NNzuG8_-wu45_5DHoJ8_TUJL3QXrYUFhPRUGdYGyNn1J6phNjv6cknveqNsr9xVLhVPZR09qnv4-TLOEjd7K3zWQuUrMMQC3zwFoxmBgcroB3HBR8fMsNDNt89nqQRDbHEiEHsMHhjEVFjLO8DwAV2kBq2c19MXRgiwkXqQuh7t2Umg-r4iQLR6RGVW5xQFYoyBLw.Hs7e4UMeZwytadcgtT5B0Ng-lOw-SOqe50SQb0Bygwc&dib_tag=se&keywords=Best%2BAI%2BLaptops&qid=1755738967&sprefix=best%2Bai%2Blaptops%2Caps%2C106&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
loader = WebBaseLoader(url)
documents = loader.load()

for doc in documents:
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print("Source:", doc.metadata.get("source", "No source metadata available"))
    print("-" * 80)
