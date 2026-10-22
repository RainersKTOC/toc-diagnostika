# TOC Finanšu Diagnostika (v3)

**Princips: Claude saprot struktūru un klasificē; Python rēķina un pārbauda.**

| Fails | Loma |
|---|---|
| `toc_ingest.py` | CSV/XLSX ielasīšana, struktūras shēma (Claude tool use), kopsummu/dublikātu/atvasināto rindu noteikšana, Σ(lapas)=gala rezultāts pārbaude |
| `toc_engine.py` | Kontu klasifikācija (Claude tool use), TOC aprēķins (S, TVC, T, OE, NP, KPI, YTD, AVG), Excel ar formulām |
| `profile_store.py` | Uzņēmuma profils — apstiprinātā klasifikācija tiek atcerēta (lokāli JSON; produkcijā nomainīt uz Supabase) |
| `app.py` | Streamlit lietotne |
| `tests/` | `run_ingest.py`, `run_engine.py`, `check_excel.py`, `test_llm_flow.py` — palaižami bez API atslēgas |

Palaišana: `pip install -r requirements.txt`, `.streamlit/secrets.toml` ar `ANTHROPIC_API_KEY = "..."`, `streamlit run app.py`.

Tests: `tests/gold.py` satur testa failu ceļus — nomainiet uz saviem. Testa failus un `profiles/` NEDRĪKST likt GitHub (`.gitignore` to jau nosedz).

Iestatījumi `toc_engine.py` augšā: `FONT_NAME`, `FONT_SIZE`, `KPI_AVG_MODE`, `GROUPS` (TOC grupu saraksts un nosaukumi tabulā).
