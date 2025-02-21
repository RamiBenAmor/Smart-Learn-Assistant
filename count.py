import fitz

def count_pdf_pages(pdf_path):
    with fitz.open(pdf_path) as doc:
        return len(doc)

# Exemple d'utilisation
pdf_file = "document.pdf"
print(f"Nombre de pages : {count_pdf_pages(pdf_file)}")
