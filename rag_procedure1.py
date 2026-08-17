article_path = '/Users/aissamhamida/Desktop/K Space.pdf'
output_folder = '/Users/aissamhamida/Desktop/yaw'

import os
import json
import pickle
import shutil
import fitz  # PyMuPDF
import numpy as np
import faiss
import torch

from PIL import Image
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

article_path = article_path.strip()
output_folder = output_folder.strip()

if not article_path:
    article_path = input("Chemin du PDF : ").strip()

if not output_folder:
    output_folder = input("Dossier de sortie : ").strip()

if not os.path.isfile(article_path):
    raise FileNotFoundError(
        f"Document introuvable : {article_path}"
    )

os.makedirs(output_folder, exist_ok=True)

PAGES_FOLDER = os.path.join(output_folder, "pages")
os.makedirs(PAGES_FOLDER, exist_ok=True)

INDEX_FOLDER = os.path.join(output_folder, "indexes")
os.makedirs(INDEX_FOLDER, exist_ok=True)

METADATA_FILE = os.path.join(
    output_folder,
    "metadata.json"
)

TEXT_INDEX_FILE = os.path.join(
    INDEX_FOLDER,
    "text_bm25.pkl"
)

VISUAL_INDEX_FILE = os.path.join(
    INDEX_FOLDER,
    "visual.faiss"
)

CONFIG_FILE = os.path.join(
    output_folder,
    "rag_config.json"
)


# VISRAG RETRIEVER


MODEL_NAME = "openbmb/VisRAG-Ret"

print("\nChargement du modèle visuel...")
print(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=(
        torch.bfloat16
        if device == "cuda"
        else torch.float32
    ),
    trust_remote_code=True
)

model.to(device)
model.eval()


# VISRAG EMBEDDING

def weighted_mean_pooling(hidden, attention_mask):

    attention_mask_ = (
        attention_mask *
        attention_mask.cumsum(dim=1)
    )

    s = torch.sum(
        hidden *
        attention_mask_.unsqueeze(-1).float(),
        dim=1
    )

    d = attention_mask_.sum(
        dim=1,
        keepdim=True
    ).float()

    return s / d


@torch.no_grad()
def encode_visual(items):

    """
    items peut contenir :

        - PIL.Image
        - texte

    VisRAG-Ret permet de représenter
    les requêtes texte et les documents
    visuels dans le même espace.
    """

    if len(items) == 0:
        return np.empty((0, 1), dtype=np.float32)

    if isinstance(items[0], str):

        inputs = {
            "text": items,
            "image": [None] * len(items),
            "tokenizer": tokenizer
        }

    else:

        inputs = {
            "text": [""] * len(items),
            "image": items,
            "tokenizer": tokenizer
        }

    inputs = {
        k: v
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    attention_mask = outputs.attention_mask

    representations = weighted_mean_pooling(
        hidden,
        attention_mask
    )

    embeddings = torch.nn.functional.normalize(
        representations,
        p=2,
        dim=1
    )

    return embeddings.cpu().float().numpy()


# OPEN PDF

print("\nOuverture du document...")

pdf = fitz.open(article_path)

number_of_pages = len(pdf)

print(
    f"Nombre de pages détectées : {number_of_pages}"
)


# DOCUMENT METADATA

metadata = []

texts = []

page_images = []




print("\nExtraction des pages...")

for page_number in range(number_of_pages):

    page = pdf[page_number]

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text = page.get_text("text")

    text = text.strip()

    texts.append(text)

    # --------------------------------------------------------
    # PAGE IMAGE
    # --------------------------------------------------------

    # 150 DPI environ
    matrix = fitz.Matrix(
        150 / 72,
        150 / 72
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image_path = os.path.join(
        PAGES_FOLDER,
        f"page_{page_number + 1:04d}.png"
    )

    pixmap.save(image_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    page_images.append(image)

    # PAGE METADATAe

    page_metadata = {

        "page_id":
            page_number,

        "page_number":
            page_number + 1,

        "image":
            os.path.relpath(
                image_path,
                output_folder
            ),

        "text":
            text,

        "text_length":
            len(text),

        "source_document":
            os.path.basename(article_path)

    }

    metadata.append(
        page_metadata
    )

    print(
        f"Page {page_number + 1}/{number_of_pages}"
    )


pdf.close()


# SAVE METADATA

print("\nSauvegarde des métadonnées...")

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


# BUILD BM25 TEXT INDEX

print("\nConstruction de l'index BM25...")

tokenized_texts = []

for text in texts:

    tokens = (
        text.lower()
        .replace("\n", " ")
        .split()
    )

    tokenized_texts.append(tokens)


bm25 = BM25Okapi(
    tokenized_texts
)


with open(
    TEXT_INDEX_FILE,
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )


# BUILD VISUAL INDEX

print("\nConstruction de l'index visuel...")

visual_embeddings = []

BATCH_SIZE = 4

for start in range(
    0,
    len(page_images),
    BATCH_SIZE
):

    batch = page_images[
        start:start + BATCH_SIZE
    ]

    embeddings = encode_visual(
        batch
    )

    visual_embeddings.append(
        embeddings
    )

    print(
        f"Embeddings : "
        f"{min(start + BATCH_SIZE, len(page_images))}"
        f"/{len(page_images)}"
    )


visual_embeddings = np.vstack(
    visual_embeddings
).astype(
    np.float32
)


# FAISS

dimension = visual_embeddings.shape[1]

print(
    "\nDimension embeddings :",
    dimension
)

# Les embeddings sont normalisés.
# Inner Product ≈ cosine similarity.

faiss_index = faiss.IndexFlatIP(
    dimension
)

faiss_index.add(
    visual_embeddings
)

faiss.write_index(
    faiss_index,
    VISUAL_INDEX_FILE
)


# SAVE CONFIGURATION

config = {

    "framework":
        "VLD-RAG-inspired multimodal RAG",

    "source_document":
        os.path.abspath(article_path),

    "number_of_pages":
        number_of_pages,

    "visual_embedding_model":
        MODEL_NAME,

    "visual_index":
        "FAISS IndexFlatIP",

    "text_index":
        "BM25",

    "page_preserving":
        True,

    "multimodal":
        True,

    "device":
        device

}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2
    )


# SAVE VISUAL EMBEDDINGS

np.save(
    os.path.join(
        INDEX_FOLDER,
        "visual_embeddings.npy"
    ),
    visual_embeddings
)


# FINAL REPORT

print("\n")
print("=" * 60)
print("       MULTIMODAL RAG CRÉÉ AVEC SUCCÈS")
print("=" * 60)

print(
    f"\nDocument : "
    f"{os.path.basename(article_path)}"
)

print(
    f"Pages : {number_of_pages}"
)

print(
    f"\nDossier RAG :\n"
    f"{os.path.abspath(output_folder)}"
)

print("\nStructure :")

print(
    f"""
{output_folder}/
│
├── pages/
│   ├── page_0001.png
│   ├── page_0002.png
│   ├── ...
│
├── indexes/
│   ├── text_bm25.pkl
│   ├── visual.faiss
│   └── visual_embeddings.npy
│
├── metadata.json
│
└── rag_config.json
"""

)

print(
    "\nLe document est maintenant indexé "
    "pour un RAG multimodal."
)

print("=" * 60)