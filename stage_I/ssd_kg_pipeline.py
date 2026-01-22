#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSD Paper → Ontology+KG → TTL
- Iterates papers (PDF/TXT/MD)
- Calls LLM with a robust hidden-reasoning prompt
- Produces per-paper JSON + TTL with evidence-backed assertions
- Validates/matches against taxonomy; adds new concepts
- Merges into a Global Knowledge Graph (.ttl) across runs (no overwrite)

Deps: pip install openai rdflib PyPDF2 python-dotenv
Env:  export OPENAI_API_KEY=...
"""

from __future__ import annotations
import os
import re
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple
from urllib.parse import quote, urlsplit, urlunsplit
import unicodedata
import tempfile
import shutil
import random


# --- Optional LLM backend (OpenAI) ---
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None  # handled gracefully

try:
    from openai import OpenAI
    from openai import (
        APIError,
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )
except Exception:
    OpenAI = None


# --- PDF extraction fallback ---
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# --- RDF ---
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS

# --------- CONFIG ---------
DEFAULT_MODEL = os.getenv("KG_LLM_MODEL", "gpt-4o")
BASE_URI = os.getenv("KG_BASE_URI", "http://example.org/ssd#")
TAXONOMY_URI = os.getenv("KG_TAXONOMY_URI", "http://example.org/ssd/taxonomy/")
ASSERTION_URI = os.getenv("KG_ASSERTION_URI", "http://example.org/ssd/assertion/")

PROMPT_PATH = os.getenv("KG_PROMPT_PATH", "prompts/ssd_cot_prompt.txt")
PROMPT_ADDENDA_PATH = os.getenv("KG_PROMPT_ADDENDA_PATH", "prompts/ssd_prompt_addenda_auto.txt")

# --------- URI MINTING TOGGLES ---------
# Use taxonomy URI directly for entities the LLM marks as "Class" (i.e., vocabulary concepts).
USE_TAXONOMY_URI_FOR_CLASS_ENTITIES = True

# For locally minted instance URIs, include the paper id and/or label?
INCLUDE_PAPER_PREFIX_IN_ENTITY_URI = False  # set to False to remove paper-title prefixes
INCLUDE_LABEL_IN_ENTITY_URI = True         # set False if you also want to hide labels in URIs

# Also type instances to their taxonomy class when we have one (rdf:type tax:...).
TYPE_INSTANCES_TO_TAXONOMY_CLASS = True


# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("ssdkg")

# --------- DEFAULT PROMPT (if no file is present) ---------
DEFAULT_PROMPT = r"""<<Put the prompt from the answer here if you don't use PROMPT_PATH>>"""

# --------- IO HELPERS ---------
def read_text_file(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def slugify_filename(name: str) -> str:
    """
    Turn an arbitrary filename (unicode, spaces, punctuation) into a safe, short slug.
    Keeps ASCII letters/digits/._- only; trims to 128 chars; guarantees non-empty.
    """
    base = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return (base[:128] or uuid.uuid4().hex)

def make_paper_id(p: Path) -> str:
    # Use a safe slug for paper_id (instead of raw stem)
    return slugify_filename(p.stem)


def read_pdf(path: Path) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. `pip install PyPDF2`")

    def _extract_text(pdf_path: Path) -> str:
        text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(text)

    try:
        # normal path
        return _extract_text(path)
    except Exception:
        # Fallback: copy to a temp file with a sanitized ASCII name and retry
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tmp = Path(tf.name)
            shutil.copyfile(path, tmp)
            return _extract_text(tmp)
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except Exception:
                    pass


def read_paper(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return read_text_file(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        return read_text_file(path)

def load_taxonomy(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_taxonomy(path: Path, taxonomy: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

# --------- PROMPT LOADING ---------
def load_prompt() -> str:
    p = Path(PROMPT_PATH)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return DEFAULT_PROMPT

def load_prompt_addenda() -> str:
    p = Path(PROMPT_ADDENDA_PATH)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""

def write_prompt_addenda(new_concepts: List[Dict[str, Any]]) -> None:
    if not new_concepts:
        return
    p = Path(PROMPT_ADDENDA_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Append (de-dup at file level)
    existing = set(p.read_text(encoding="utf-8").splitlines()) if p.exists() else set()
    lines = []
    for nc in new_concepts:
        label = nc.get("label")
        parent = nc.get("suggested_parent_path")
        if not label or not parent:
            continue
        line = f"- Prefer mapping new concept “{label}” under “{parent}”."
        if line not in existing:
            lines.append(line)
    if lines:
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

# --------- URI ENCODING HELPERS ---------
def _encode_path_segments(path: str) -> str:
    segs = [quote(seg, safe="") for seg in (path or "").strip("/").split("/")]
    return "/".join(segs)

def _join_base_and_path(base: str, path: str) -> str:
    base = base.rstrip("/")
    enc = _encode_path_segments(path)
    return f"{base}/{enc}" if enc else base

def path_to_uri(path_str: str) -> str:
    if re.match(r"^https?://", path_str or ""):
        parts = urlsplit(path_str)
        safe_path = _encode_path_segments(parts.path)
        return urlunsplit((parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment))
    return _join_base_and_path(TAXONOMY_URI, path_str or "")

def flatten_taxonomy_paths(taxonomy: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    def walk(node: Any, trail: List[str]):
        if isinstance(node, dict):
            for k, v in node.items():
                new_trail = trail + [k]
                mapping[k] = "/".join(new_trail)
                walk(v, new_trail)
        elif isinstance(node, list):
            for v in node:
                mapping[v] = "/".join(trail + [v])
    walk(taxonomy, [])
    return {k: _join_base_and_path(TAXONOMY_URI, v) for k, v in mapping.items()}

# --------- LLM BACKEND ---------
class LLMBackend:
    def __init__(self, model: str = DEFAULT_MODEL):
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI client not available. Install `openai`>=1.0 and set OPENAI_API_KEY."
            )
        self.client = OpenAI()
        self.model = model
        self.base_prompt = load_prompt()

    def run(self, full_paper_text: str, taxonomy_json: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        sys_prompt = self.base_prompt
        addenda = load_prompt_addenda()

        messages = [{"role": "system", "content": sys_prompt}]
        if addenda.strip():
            messages.append({"role": "system", "content": "Additional mapping notes:\n" + addenda})
        messages.append({
            "role": "user",
            "content": json.dumps({"full_paper_text": full_paper_text, "taxonomy": taxonomy_json})
        })

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=60.0,  # seconds
                )
                content = resp.choices[0].message.content
                return json.loads(content)
            except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, APIError) as e:
                last_err = e
                # exponential backoff with jitter
                wait = min(2 ** attempt, 60) + random.uniform(0, 0.5)
                log.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s")
                if attempt == max_retries:
                    break
                time.sleep(wait)
            except json.JSONDecodeError as e:
                # Should be rare because we use response_format=json_object, but guard anyway
                last_err = e
                log.warning(f"JSON parse error (attempt {attempt}/{max_retries}): {e}. Retrying with backoff.")
                wait = min(2 ** attempt, 60) + random.uniform(0, 0.5)
                if attempt == max_retries:
                    break
                time.sleep(wait)

        # If we get here, all retries failed
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")

# --------- JSON SANITY ---------
def coerce_json(obj: Any) -> Dict[str, Any]:
    keys = {"paper_id", "entities", "triples", "axioms", "mappings", "new_concepts"}
    if not isinstance(obj, dict) or not keys.issubset(set(obj.keys())):
        raise ValueError("LLM JSON missing required keys.")
    return obj

# --------- VALIDATION / MAPPING ---------
def validate_and_map(
    result: Dict[str, Any],
    tax_lookup: Dict[str, str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    entities = result.get("entities", [])
    label_by_id = {e["id"]: e.get("label") for e in entities}

    # Normalize & attach taxonomy URIs
    for e in entities:
        tp = e.get("taxonomy_path")
        turi = e.get("taxonomy_uri")
        if tp:
            e["taxonomy_uri"] = path_to_uri(tp)
        elif turi:
            e["taxonomy_uri"] = path_to_uri(turi)
        else:
            label = e.get("label")
            if label and label in tax_lookup:
                e["taxonomy_uri"] = path_to_uri(tax_lookup[label])
                e["taxonomy_path"] = e.get("taxonomy_path") or e["taxonomy_uri"].replace(TAXONOMY_URI, "")
            else:
                e["taxonomy_uri"] = None

    # Filter/clean triples: require evidence and a predicate
    clean_triples = []
    for t in result.get("triples", []):
        if not t.get("evidence") or not t.get("p"):
            continue

        # Heuristic: fix contradictory verbs based on evidence wording
        ev = t.get("evidence", "").lower()
        if t["p"] == "degrades" and any(w in ev for w in ["improve", "decrease", "reduced", "lower"]):
            t["p"] = "improves"
        elif t["p"] == "improves" and any(w in ev for w in ["increase", "worse", "degrade", "higher"]):
            t["p"] = "degrades"

        # Keep entity->entity triples or entity->literal triples
        obj = t.get("o")
        if isinstance(obj, dict) and "@value" in obj:
            if t.get("s") in label_by_id:
                clean_triples.append(t)
        elif t.get("s") in label_by_id and t.get("o") in label_by_id:
            clean_triples.append(t)

    result["triples"] = clean_triples

    # De-dup new concepts
    proposals = result.get("new_concepts", []) or []
    seen = set()
    uniq_new = []
    for nc in proposals:
        key = (nc.get("label"), nc.get("suggested_parent_path"))
        if key not in seen and nc.get("label") and nc.get("suggested_parent_path"):
            uniq_new.append(nc)
            seen.add(key)

    # Keep entities that are referenced or have taxonomy mapping
    referenced_ids = {t["s"] for t in clean_triples if isinstance(t.get("s"), str)}
    referenced_ids |= {t["o"] for t in clean_triples if isinstance(t.get("o"), str)}
    result["entities"] = [e for e in entities if e["id"] in referenced_ids or e.get("taxonomy_uri")]

    return result, uniq_new

# --------- TAXONOMY UPDATE ---------
def insert_new_concepts(taxonomy: Dict[str, Any], proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get_node_by_path(path_list: List[str]) -> Any:
        node = taxonomy
        for part in path_list:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    for p in proposals:
        label = p.get("label")
        parent_path = p.get("suggested_parent_path")
        if not label or not parent_path:
            continue
        parts = parent_path.split("/")
        parent = get_node_by_path(parts)
        if isinstance(parent, dict):
            parent.setdefault(label, [])
        elif isinstance(parent, list):
            if label not in parent:
                parent.append(label)
        else:
            # fallback: attach under top-level SSD if available
            if "SSD" in taxonomy and isinstance(taxonomy["SSD"], dict):
                taxonomy["SSD"].setdefault(label, [])
    return taxonomy

# --------- RDF / TTL ---------
@dataclass
class KGContext:
    base: Namespace = field(default_factory=lambda: Namespace(BASE_URI))
    tax: Namespace = field(default_factory=lambda: Namespace(TAXONOMY_URI))
    asn: Namespace = field(default_factory=lambda: Namespace(ASSERTION_URI))

def uri_safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def build_graph(result: Dict[str, Any]) -> Graph:
    g = Graph()
    NS = KGContext()
    g.bind("ssd", NS.base)
    g.bind("tax", NS.tax)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dct", DCTERMS)

    # seed properties (unchanged)
    obj_props = {
        "operatesUnder", "hasEnvironment", "hasExternalFactor", "hasWorkloadProfile",
        "impacts", "improves", "degrades", "correlatesWith",
        "subject", "predicate", "object", "hasDeviceType"
    }
    dt_props = {
        "hasTemperature", "hasRelativeHumidity", "improvedBy", "degradedBy",
        "percentile", "unit", "confidence", "hasAccessPattern",
        "hasReadWriteMix", "hasBlockSize", "hasQueueDepth"
    }
    for pn in obj_props:
        g.add((NS.base[pn], RDF.type, OWL.ObjectProperty))
    for pn in dt_props:
        g.add((NS.base[pn], RDF.type, OWL.DatatypeProperty))

    # ---------- NEW: helper to mint or reuse URIs ----------
    def _entity_uri(e: Dict[str, Any]) -> URIRef:
        e_type = (e.get("type") or "").lower()  # "class" or "instance"
        turi = e.get("taxonomy_uri")
        if USE_TAXONOMY_URI_FOR_CLASS_ENTITIES and e_type == "class" and turi:
            return URIRef(path_to_uri(turi))

        # Otherwise mint a local, paper-agnostic instance URI
        parts = []
        if INCLUDE_PAPER_PREFIX_IN_ENTITY_URI:
            parts.append(result["paper_id"])
        parts.append(e["id"])
        if INCLUDE_LABEL_IN_ENTITY_URI:
            parts.append(e["label"])
        return URIRef(NS.base[uri_safe("_".join(parts))])

    # Entities
    id2uri: Dict[str, URIRef] = {}
    for e in result.get("entities", []):
        eid = e["id"]
        euri = _entity_uri(e)
        id2uri[eid] = euri

        # If we reused a taxonomy URI for a Class, we shouldn't re-state it's an instance of Thing.
        if USE_TAXONOMY_URI_FOR_CLASS_ENTITIES and (e.get("type","").lower() == "class") and e.get("taxonomy_uri"):
            # Optionally annotate the class with a label if missing; generally taxonomy already has one.
            g.add((euri, RDFS.label, Literal(e.get("label"))))
        else:
            # local instance node
            g.add((euri, RDF.type, OWL.Thing))
            g.add((euri, RDFS.label, Literal(e.get("label"))))
            # If we know its taxonomy class, type it
            if TYPE_INSTANCES_TO_TAXONOMY_CLASS and e.get("taxonomy_uri"):
                g.add((euri, RDF.type, URIRef(path_to_uri(e["taxonomy_uri"]))))

        # Keep provenance pointer to the taxonomy resource if present
        if e.get("taxonomy_uri"):
            g.add((euri, RDFS.isDefinedBy, URIRef(path_to_uri(e["taxonomy_uri"]))))

    # Axioms (unchanged)
    for ax in result.get("axioms", []):
        axuri = URIRef(NS.base[uri_safe(f"axiom_{uuid.uuid4().hex[:8]}")])
        g.add((axuri, RDF.type, OWL.Axiom))
        g.add((axuri, RDFS.comment, Literal(ax)))

    # Triples + evidence (reified) — unchanged except it now works with reused taxonomy URIs
    for t in result.get("triples", []):
        s = id2uri.get(t.get("s"))
        p_local = t.get("p")
        if not s or not p_local:
            continue
        pred = URIRef(NS.base[uri_safe(p_local)])
        obj = t.get("o")
        ev_text = t.get("evidence", "")
        try:
            conf = float(t.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        asn = URIRef(NS.asn[uuid.uuid4().hex])
        g.add((asn, RDF.type, NS.base["Assertion"]))
        g.add((asn, NS.base["subject"], s))
        g.add((asn, NS.base["predicate"], pred))
        g.add((asn, DCTERMS.source, Literal(ev_text)))
        g.add((asn, NS.base["confidence"], Literal(conf, datatype=XSD.decimal)))

        if isinstance(obj, dict) and "@value" in obj:
            lit = Literal(obj["@value"])
            g.add((s, pred, lit))
            g.add((asn, NS.base["objectLiteral"], lit))
            if "unit" in obj and obj["unit"] is not None:
                g.add((s, NS.base["unit"], Literal(obj["unit"])))
        else:
            o_uri = id2uri.get(obj)
            if not o_uri:
                continue
            g.add((s, pred, o_uri))
            g.add((asn, NS.base["object"], o_uri))

    return g


# --------- PIPELINE ---------
def process_paper(
    paper_path: Path,
    taxonomy_json: Dict[str, Any],
    tax_lookup: Dict[str, str],
    llm: LLMBackend,
    out_dir: Path
) -> Tuple[Graph, Dict[str, Any], List[Dict[str, Any]]]:
    log.info(f"Reading paper: {paper_path.name}")
    text = read_paper(paper_path)

    log.info("Calling LLM…")
    result_raw = llm.run(text, taxonomy_json)
    result = coerce_json(result_raw)

    # result["paper_id"] = result.get("paper_id") or paper_path.stem
    result["paper_id"] = result.get("paper_id") or make_paper_id(paper_path)
    result["paper_original_filename"] = paper_path.name  # keep original for provenance

    safe_stem = slugify_filename(paper_path.stem)  # for output files


    log.info("Validating & mapping…")
    clean_result, new_concepts = validate_and_map(result, tax_lookup)

    log.info("Building RDF graph…")
    g = build_graph(clean_result)

    # Save per-paper TTL + JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    # ttl_path = out_dir / f"{paper_path.stem}.ttl"
    ttl_path = out_dir / f"{safe_stem}.ttl"
    g.serialize(destination=str(ttl_path), format="turtle")
    log.info(f"Wrote per-paper TTL: {ttl_path}")

    # json_path = out_dir / f"{paper_path.stem}.kg.json"
    json_path = out_dir / f"{safe_stem}.kg.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_result, f, indent=2, ensure_ascii=False)

    return g, clean_result, new_concepts

def merge_graphs(graphs: List[Graph]) -> Graph:
    g_all = Graph()
    for g in graphs:
        for triple in g:
            g_all.add(triple)
    return g_all

def merge_with_existing_global(global_ttl_path: Path, g_new: Graph) -> Graph:
    """
    Load existing global TTL if present, add its triples, then write back with new ones.
    """
    g_all = Graph()
    if global_ttl_path.exists():
        try:
            g_all.parse(str(global_ttl_path), format="turtle")
            log.info(f"Loaded existing global graph: {global_ttl_path}")
        except Exception as e:
            log.warning(f"Could not parse existing global TTL, starting fresh: {e}")
    # add new triples
    for t in g_new:
        g_all.add(t)
    return g_all

# --------- CLI ---------
def main(
    papers_dir: str,
    taxonomy_path: str,
    out_dir: str,
    model: str = DEFAULT_MODEL
):
    papers_dir = Path(papers_dir)
    taxonomy_path = Path(taxonomy_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_json = load_taxonomy(taxonomy_path)
    tax_lookup = flatten_taxonomy_paths(taxonomy_json)

    llm = LLMBackend(model=model)

    # Collect papers
    paper_paths: List[Path] = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        paper_paths.extend(sorted(papers_dir.glob(ext)))
    if not paper_paths:
        log.warning("No papers found (supported: .pdf, .txt, .md)")
        return

    graphs: List[Graph] = []
    all_new_concepts: List[Dict[str, Any]] = []

    for p in paper_paths:
        try:
            g, _, proposals = process_paper(p, taxonomy_json, tax_lookup, llm, out_dir)
            graphs.append(g)
            all_new_concepts.extend(proposals or [])
            time.sleep(0.4)  # polite pacing
        except Exception as e:
            log.exception(f"Failed on {p.name}: {e}")

    # Merge current-run graphs
    if graphs:
        g_run = merge_graphs(graphs)
        global_ttl = out_dir / "global_knowledge_graph.ttl"
        # Merge with previous global if exists
        g_all = merge_with_existing_global(global_ttl, g_run)
        g_all.serialize(destination=str(global_ttl), format="turtle")
        log.info(f"Updated GLOBAL TTL: {global_ttl}")

    # Update taxonomy with any new concepts
    if all_new_concepts:
        log.info(f"Adding {len(all_new_concepts)} proposed concept(s) to taxonomy…")
        updated = insert_new_concepts(taxonomy_json, all_new_concepts)
        save_taxonomy(taxonomy_path, updated)
        log.info(f"Updated taxonomy JSON saved to: {taxonomy_path}")
        # (Optional) augment prompt addenda for future runs
        write_prompt_addenda(all_new_concepts)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SSD KG Builder")
    ap.add_argument("--papers_dir", required=True, help="Folder with papers (.pdf/.txt/.md)")
    ap.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    ap.add_argument("--out_dir", required=True, help="Output folder for TTL and JSON")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name (e.g., gpt-4o)")
    args = ap.parse_args()
    main(args.papers_dir, args.taxonomy, args.out_dir, args.model)
