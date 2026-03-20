import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
from huggingface_hub import snapshot_download

COULEURS = {
    "LOC": "#a8d5e2",
    "ARC": "#f4a261"
}

@st.cache_resource
def charger_modele():
    chemin = snapshot_download(repo_id="VictoriaMmm/ner-helleniques")
    return spacy.load(chemin)

nlp = charger_modele()

st.title("🏛️ Détection d'entités nommées")
st.caption("Modèle entraîné sur les Correspondances Helléniques — étiquettes : LOC, ARC")

st.subheader("📥 Saisie du texte")
mode = st.radio("Mode de saisie", ["Texte brut", "Fichier CoNLL"], horizontal=True)

texte = ""

if mode == "Texte brut":
    texte = st.text_area("Collez votre texte ici", height=200)

else:
    fichier = st.file_uploader("Uploadez un fichier CoNLL", type=["conll", "txt"])
    if fichier:
        tokens = []
        for ligne in fichier.read().decode("utf-8").splitlines():
            ligne = ligne.strip()
            if ligne == "" or ligne.startswith("#"):
                continue
            parties = ligne.split()
            if len(parties) >= 1:
                tokens.append(parties[0])
        texte = " ".join(tokens)
        st.success(f"{len(tokens)} tokens chargés.")
        st.text_area("Texte reconstruit", texte, height=150)

if texte and st.button("🔍 Analyser"):
    doc = nlp(texte)

    st.subheader("📝 Entités surlignées")
    html = displacy.render(doc, style="ent", options={"colors": COULEURS}, page=False)
    st.html(html)

    st.subheader("📊 Tableau des entités détectées")
    if doc.ents:
        df = pd.DataFrame([
            {"Entité": ent.text, "Étiquette": ent.label_, "Début": ent.start_char, "Fin": ent.end_char}
            for ent in doc.ents
        ])
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Comptage par étiquette")
        comptage = df["Étiquette"].value_counts().reset_index()
        comptage.columns = ["Étiquette", "Occurrences"]
        st.dataframe(comptage, use_container_width=True)

        st.subheader("⬇️ Télécharger les résultats en CoNLL")
        ent_map = {}
        for ent in doc.ents:
            for i, token in enumerate(ent):
                prefix = "B" if i == 0 else "I"
                ent_map[token.i] = f"{prefix}-{ent.label_}"

        lignes_conll = []
        for token in doc:
            etiquette = ent_map.get(token.i, "O")
            lignes_conll.append(f"{token.text} {etiquette}")

        contenu_conll = "\n".join(lignes_conll)
        st.download_button(
            label="💾 Télécharger le fichier CoNLL",
            data=contenu_conll.encode("utf-8"),
            file_name="resultats_NER.conll",
            mime="text/plain"
        )
    else:
        st.info("Aucune entité détectée dans ce texte.")
