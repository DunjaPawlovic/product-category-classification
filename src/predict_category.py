import pickle

# load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

while True:
    text = input("Unesi naziv proizvoda (ili 'exit'): ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])
    pred = model.predict(vec)

    print("Kategorija:", pred[0])
