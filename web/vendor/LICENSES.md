# Bibliothèques embarquées

KIRA n'utilise aucun CDN : tout ce que l'interface charge vient de ce dossier.

| Fichier | Projet | Licence |
|---|---|---|
| `marked.umd.js` | marked (rendu Markdown) | MIT — `LICENSE-marked` |
| `purify.min.js` | DOMPurify (nettoyage du HTML) | Apache-2.0 / MPL-2.0 — `LICENSE-dompurify` |
| `katex.min.js`, `katex.min.css`, `fonts/KaTeX_*.woff2` | KaTeX (formules) | MIT — `LICENSE-katex` |
| `fonts/lora.woff`, `fonts/lora-italic.woff` | Lora, sous-ensemble latin (© Cyreal) | SIL Open Font License 1.1 |

Pour mettre à jour : `npm pack marked dompurify katex` puis recopier les fichiers ci-dessus.
