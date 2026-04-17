# Klasifikacija kategorija proizvoda (ML projekat)

# Opis projekta
Ovaj projekat koristi mašinsko učenje za automatsku klasifikaciju proizvoda na osnovu njihovog naziva.

Model uči iz postojećih podataka i predviđa kojoj kategoriji novi proizvod pripada.

---

# Skup podataka
Dataset sadrži informacije o proizvodima:

- Product Title (naziv proizvoda)
- Category Label (kategorija – ciljna promenljiva)
- Merchant ID
- Product ID
- Number of Views
- Merchant Rating
- Listing Date

---

# Korišćene tehnologije
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Google Colab

---

# Kako radi model
1. Učitavanje i čišćenje podataka  
2. Obrada teksta (TF-IDF)  
3. Podela na trening i test skup  
4. Treniranje modela (Logistic Regression)  
5. Evaluacija performansi modela  

---

# Rezultati
Model postiže oko **94% tačnosti** na test skupu podataka.

---

# Struktura projekta
# product-category-classification
data/
notebook/
src/
models/
README.md

---

# Pokretanje projekta
# Treniranje modela
python src/train_model.py
Sačuvani fajlovi modela
model.pkl (trenirani model)
vectorizer.pkl (TF-IDF vektorizator)
# Primer korišćenja
Ulaz:
iPhone 13 128GB
Izlaz:
Mobile Phones
