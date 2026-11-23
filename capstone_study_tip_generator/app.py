"""Interactive Demonstration Interface: Domain-Adapted Study Tip Generator.

Execution:
  python -m streamlit run app.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

# Safe optional import of PyTorch and Transformers to ensure resilience across
# environments with strict Windows Application Control policies or CPU-only setups.
HAS_TORCH: bool = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except (ImportError, OSError, Exception):
    HAS_TORCH = False
    torch = None

# -----------------------------------------------------------------------------
# Base Directories & Knowledge Base Ingestion
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_FOLDER: Path = BASE_DIR / "lab_outputs" / "study_tip_gpt2"
DATA_DIR: Path = BASE_DIR / "data"
OUTPUTS_PATH: Path = BASE_DIR / "lab_outputs" / "outputs.json"
TRAIN_PATH: Path = DATA_DIR / "train.json"

# Embedded avatar graphic for Vinayak Sharma (matches floating dock profile)
AVATAR_B64: str = (
    "/9j/2wCEAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0"
    "Hyc5PTgyPC4zNDIBCQkJDAsMGA0NGDIhHCEyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMv/AABEIAcwBzAMBIgACEQEDEQH/xAGiAAABBQEBAQEBAQAAAAAAAAAAAQID"
    "BAUGBwgJCgsQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNi"
    "coIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeI"
    "iYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz"
    "9PX29/j5+gEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoLEQACAQIEBAMEBwUEBAABAncAAQIDEQQF"
    "ITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpT"
    "VFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrC"
    "w8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/APAKKKKACiiigAoo"
    "ooAWikoFAC0UUUAFLSUUALRRRQAClpBS0AFLRjNKKAEFSxpvPFRjrUySFBx1NAEjLtwBxTwijk9PSogW"
    "c5qUggYA+poAcrY5AAHvSNKzHagNR4XjLEn0FWIyRwBx6DigBiwdTJx+NOCgEbAQKkcxjpgfXmq5DsTt"
    "RjQBKCmDmQiopHAXG0EevQ0nlscbjt+hqFkXdxz9TQAMV+v9KQIx6Umxj0Q1Ihkj+8pA7dqAIjE5OCG9"
    "qY0bBsEEH6VeS4CjaTj6Ukkgk/j5+tAFTy3wOMg00pzyMVYyxGM8etRFSc8c+lADNo9aBjPtQTxSZoAc"
    "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"
    "qAOYAAoPWlxSUAf/2Q=="
)


@st.cache_data(show_spinner=False)
def load_pedagogical_knowledge_base() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[Dict]]:
    """Loads benchmark outputs and training pairs into a fast, normalized lookup engine."""
    knowledge_map: Dict[str, str] = {}
    comparative_map: Dict[str, Dict[str, str]] = {}
    benchmark_list: List[Dict] = []

    # Load 40 training instructional pairs
    if TRAIN_PATH.exists():
        try:
            with open(TRAIN_PATH, "r", encoding="utf-8") as f:
                train_data = json.load(f)
                for topic, tip in train_data:
                    knowledge_map[topic.strip().lower()] = tip
        except Exception:
            pass

    # Load 12 held-out benchmark comparative predictions
    if OUTPUTS_PATH.exists():
        try:
            with open(OUTPUTS_PATH, "r", encoding="utf-8") as f:
                benchmark_list = json.load(f)
                for item in benchmark_list:
                    t_key = item.get("topic", "").strip().lower()
                    if t_key:
                        knowledge_map[t_key] = item.get("adapted", "")
                        comparative_map[t_key] = {
                            "base": item.get("base", ""),
                            "adapted": item.get("adapted", ""),
                        }
        except Exception:
            pass

    return knowledge_map, comparative_map, benchmark_list


# -----------------------------------------------------------------------------
# Pedagogical Topic Diagram Generator (Vector SVG in Base64)
# -----------------------------------------------------------------------------
def get_topic_diagram_uri(topic: str) -> str:
    """Generates a topic-specific vector diagram illustrating the concept."""
    topic_clean = topic.strip().lower()

    if "recursion" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Recursion Call Tree: solve(3) → solve(2) + solve(1)</text>
  <circle cx="270" cy="65" r="22" fill="#059669"/>
  <text x="270" y="70" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">f(3)</text>
  <line x1="254" y1="80" x2="160" y2="120" stroke="#059669" stroke-width="2"/>
  <line x1="286" y1="80" x2="380" y2="120" stroke="#059669" stroke-width="2"/>
  <circle cx="160" cy="130" r="18" fill="#10B981"/>
  <text x="160" y="135" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">f(2)</text>
  <circle cx="380" cy="130" r="18" fill="#10B981"/>
  <text x="380" y="135" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">f(1)</text>
  <line x1="148" y1="144" x2="105" y2="180" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="3,3"/>
  <line x1="172" y1="144" x2="215" y2="180" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="3,3"/>
  <circle cx="105" cy="188" r="14" fill="#D1FAE5" stroke="#059669" stroke-width="1.5"/>
  <text x="105" y="192" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#065F46" text-anchor="middle">f(1)</text>
  <circle cx="215" cy="188" r="14" fill="#D1FAE5" stroke="#059669" stroke-width="1.5"/>
  <text x="215" y="192" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#065F46" text-anchor="middle">f(0)</text>
  <rect x="330" y="175" width="120" height="26" fill="#ECFDF5" stroke="#A7F3D0" rx="4"/>
  <text x="390" y="192" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#059669" text-anchor="middle">Base Cases Return</text>
</svg>"""
    elif "sql" in topic_clean or "join" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Relational SQL Joins: Table A ⋈ Table B</text>
  <circle cx="205" cy="115" r="65" fill="#E0F2FE" stroke="#0284C7" stroke-width="2" fill-opacity="0.6"/>
  <text x="170" y="120" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#0369A1">Table A (Left)</text>
  <circle cx="335" cy="115" r="65" fill="#FEF3C7" stroke="#D97706" stroke-width="2" fill-opacity="0.6"/>
  <text x="370" y="120" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#B45309">Table B (Right)</text>
  <path d="M 270,60 A 65 65 0 0 1 270,170 A 65 65 0 0 1 270,60" fill="#059669" fill-opacity="0.85"/>
  <text x="270" y="112" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">INNER JOIN</text>
  <text x="270" y="128" font-family="system-ui, sans-serif" font-size="9" fill="#FFFFFF" text-anchor="middle">A.id = B.cust_id</text>
  <rect x="70" y="188" width="400" height="22" fill="#FFFFFF" stroke="#E5E9E7" rx="4"/>
  <text x="270" y="203" font-family="system-ui, sans-serif" font-size="10" font-weight="500" fill="#374151" text-anchor="middle">Filter matched rows on foreign keys before projecting query columns</text>
</svg>"""
    elif "search" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Binary Search: O(log n) Divide-and-Conquer</text>
  <g transform="translate(45, 70)">
    <rect x="0" y="0" width="55" height="45" fill="#F3F4F6" stroke="#D1D5DB" rx="4"/>
    <text x="27" y="27" font-family="system-ui, sans-serif" font-size="12" fill="#6B7280" text-anchor="middle">4</text>
    <rect x="58" y="0" width="55" height="45" fill="#F3F4F6" stroke="#D1D5DB" rx="4"/>
    <text x="85" y="27" font-family="system-ui, sans-serif" font-size="12" fill="#6B7280" text-anchor="middle">12</text>
    <rect x="116" y="0" width="55" height="45" fill="#F3F4F6" stroke="#D1D5DB" rx="4"/>
    <text x="143" y="27" font-family="system-ui, sans-serif" font-size="12" fill="#6B7280" text-anchor="middle">23</text>
    <rect x="174" y="0" width="55" height="45" fill="#FEF3C7" stroke="#F59E0B" stroke-width="2" rx="4"/>
    <text x="201" y="27" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#B45309" text-anchor="middle">38</text>
    <rect x="232" y="0" width="55" height="45" fill="#D1FAE5" stroke="#059669" stroke-width="2" rx="4"/>
    <text x="259" y="27" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#065F46" text-anchor="middle">56</text>
    <rect x="290" y="0" width="55" height="45" fill="#D1FAE5" stroke="#059669" stroke-width="2" rx="4"/>
    <text x="317" y="27" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#065F46" text-anchor="middle">72</text>
    <rect x="348" y="0" width="55" height="45" fill="#D1FAE5" stroke="#059669" stroke-width="2" rx="4"/>
    <text x="375" y="27" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#065F46" text-anchor="middle">91</text>
  </g>
  <text x="72" y="140" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#6B7280" text-anchor="middle">↑ Low (0)</text>
  <text x="246" y="140" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#B45309" text-anchor="middle">↑ Mid (3)</text>
  <text x="420" y="140" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#6B7280" text-anchor="middle">↑ High (6)</text>
  <rect x="70" y="165" width="400" height="30" fill="#ECFDF5" stroke="#A7F3D0" rx="4"/>
  <text x="270" y="184" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#059669" text-anchor="middle">Target 72 > 38 → Discard left half, set Low = Mid + 1</text>
</svg>"""
    elif "list" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Singly Linked List: Pointer Chain</text>
  <rect x="40" y="80" width="60" height="45" fill="#059669" rx="4"/>
  <text x="70" y="108" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">10</text>
  <rect x="100" y="80" width="30" height="45" fill="#047857" rx="2"/>
  <circle cx="115" cy="102" r="4" fill="#FFFFFF"/>
  <line x1="115" y1="102" x2="165" y2="102" stroke="#059669" stroke-width="2.5"/>
  <rect x="175" y="80" width="60" height="45" fill="#059669" rx="4"/>
  <text x="205" y="108" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">20</text>
  <rect x="235" y="80" width="30" height="45" fill="#047857" rx="2"/>
  <circle cx="250" cy="102" r="4" fill="#FFFFFF"/>
  <line x1="250" y1="102" x2="300" y2="102" stroke="#059669" stroke-width="2.5"/>
  <rect x="310" y="80" width="60" height="45" fill="#059669" rx="4"/>
  <text x="340" y="108" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#FFFFFF" text-anchor="middle">30</text>
  <rect x="370" y="80" width="30" height="45" fill="#047857" rx="2"/>
  <text x="385" y="108" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#D1FAE5" text-anchor="middle">∅</text>
  <text x="85" y="65" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#065F46" text-anchor="middle">HEAD</text>
  <text x="355" y="65" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#6B7280" text-anchor="middle">TAIL (NULL)</text>
  <rect x="70" y="155" width="400" height="30" fill="#ECFDF5" stroke="#A7F3D0" rx="4"/>
  <text x="270" y="174" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#059669" text-anchor="middle">Sequential traversal: current = current.next until NULL is encountered</text>
</svg>"""
    elif "hash" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Hash Table: Key → Hash Function → Bucket Array O(1)</text>
  <rect x="40" y="60" width="90" height="32" fill="#E0F2FE" stroke="#0284C7" rx="4"/>
  <text x="85" y="81" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#0369A1" text-anchor="middle">"alice"</text>
  <rect x="40" y="105" width="90" height="32" fill="#E0F2FE" stroke="#0284C7" rx="4"/>
  <text x="85" y="126" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#0369A1" text-anchor="middle">"bob"</text>
  <rect x="180" y="70" width="95" height="60" fill="#059669" rx="6"/>
  <text x="227" y="97" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">Hash Func</text>
  <text x="227" y="115" font-family="system-ui, sans-serif" font-size="9" fill="#D1FAE5" text-anchor="middle">h(key) % N</text>
  <line x1="130" y1="76" x2="180" y2="90" stroke="#059669" stroke-width="2"/>
  <line x1="130" y1="121" x2="180" y2="110" stroke="#059669" stroke-width="2"/>
  <rect x="330" y="55" width="160" height="26" fill="#FFFFFF" stroke="#D1D5DB" rx="3"/>
  <text x="350" y="72" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#6B7280">[0]</text>
  <rect x="330" y="85" width="160" height="26" fill="#D1FAE5" stroke="#059669" stroke-width="1.5" rx="3"/>
  <text x="350" y="102" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#065F46">[1] "alice" → 95</text>
  <rect x="330" y="115" width="160" height="26" fill="#FFFFFF" stroke="#D1D5DB" rx="3"/>
  <text x="350" y="132" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#6B7280">[2]</text>
  <rect x="330" y="145" width="160" height="26" fill="#D1FAE5" stroke="#059669" stroke-width="1.5" rx="3"/>
  <text x="350" y="162" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#065F46">[3] "bob" → 82</text>
  <line x1="275" y1="95" x2="330" y2="98" stroke="#059669" stroke-width="2"/>
  <line x1="275" y1="115" x2="330" y2="158" stroke="#059669" stroke-width="2"/>
</svg>"""
    elif "branch" in topic_clean or "git" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Git Version Control: Branching and Fast-Forward Merge</text>
  <line x1="60" y1="120" x2="480" y2="120" stroke="#059669" stroke-width="3"/>
  <circle cx="90" cy="120" r="14" fill="#059669"/>
  <text x="90" y="124" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#FFFFFF" text-anchor="middle">C1</text>
  <circle cx="180" cy="120" r="14" fill="#059669"/>
  <text x="180" y="124" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#FFFFFF" text-anchor="middle">C2</text>
  <circle cx="450" cy="120" r="16" fill="#047857" stroke="#10B981" stroke-width="2"/>
  <text x="450" y="124" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#FFFFFF" text-anchor="middle">C5</text>
  <path d="M 180,120 C 230,60 270,60 310,60 L 390,60 C 420,60 435,95 450,120" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-dasharray="4,2"/>
  <circle cx="280" cy="60" r="13" fill="#F59E0B"/>
  <text x="280" y="64" font-family="system-ui, sans-serif" font-size="9" font-weight="700" fill="#FFFFFF" text-anchor="middle">C3</text>
  <circle cx="360" cy="60" r="13" fill="#F59E0B"/>
  <text x="360" y="64" font-family="system-ui, sans-serif" font-size="9" font-weight="700" fill="#FFFFFF" text-anchor="middle">C4</text>
  <rect x="420" y="150" width="60" height="20" fill="#D1FAE5" stroke="#059669" rx="3"/>
  <text x="450" y="164" font-family="system-ui, sans-serif" font-size="9" font-weight="700" fill="#065F46" text-anchor="middle">main</text>
  <rect x="330" y="24" width="60" height="20" fill="#FEF3C7" stroke="#F59E0B" rx="3"/>
  <text x="360" y="38" font-family="system-ui, sans-serif" font-size="9" font-weight="700" fill="#B45309" text-anchor="middle">feature</text>
  <text x="270" y="200" font-family="system-ui, sans-serif" font-size="11" fill="#4B5563" text-anchor="middle">git checkout -b feature → commit changes → git merge main</text>
</svg>"""
    elif "big o" in topic_clean or "notation" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Asymptotic Complexity: Big O Growth Curves</text>
  <line x1="70" y1="180" x2="490" y2="180" stroke="#374151" stroke-width="2"/>
  <line x1="70" y1="180" x2="70" y2="50" stroke="#374151" stroke-width="2"/>
  <text x="480" y="196" font-family="system-ui, sans-serif" font-size="10" fill="#6B7280">Elements (n)</text>
  <text x="40" y="55" font-family="system-ui, sans-serif" font-size="10" fill="#6B7280">Time</text>
  <line x1="70" y1="165" x2="480" y2="165" stroke="#10B981" stroke-width="2.5"/>
  <text x="490" y="168" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#059669">O(1)</text>
  <path d="M 70,180 Q 200,150 480,140" fill="none" stroke="#059669" stroke-width="2.5"/>
  <text x="490" y="142" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#059669">O(log n)</text>
  <line x1="70" y1="180" x2="460" y2="90" stroke="#3B82F6" stroke-width="2.5"/>
  <text x="470" y="92" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#2563EB">O(n)</text>
  <path d="M 70,180 Q 180,170 240,55" fill="none" stroke="#EF4444" stroke-width="2.5"/>
  <text x="250" y="58" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#DC2626">O(n²)</text>
</svg>"""
    elif "stack" in topic_clean or "queue" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Stack (LIFO) vs Queue (FIFO) Mechanics</text>
  <g transform="translate(60, 50)">
    <rect x="0" y="0" width="180" height="130" fill="#FFFFFF" stroke="#E5E9E7" rx="6"/>
    <text x="90" y="24" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#059669" text-anchor="middle">Stack (LIFO)</text>
    <rect x="35" y="90" width="110" height="22" fill="#E0F2FE" stroke="#0284C7" rx="3"/>
    <text x="90" y="105" font-family="system-ui, sans-serif" font-size="10" fill="#0369A1" text-anchor="middle">Item 1 (Bottom)</text>
    <rect x="35" y="64" width="110" height="22" fill="#E0F2FE" stroke="#0284C7" rx="3"/>
    <text x="90" y="79" font-family="system-ui, sans-serif" font-size="10" fill="#0369A1" text-anchor="middle">Item 2</text>
    <rect x="35" y="38" width="110" height="22" fill="#D1FAE5" stroke="#059669" stroke-width="1.5" rx="3"/>
    <text x="90" y="53" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#065F46" text-anchor="middle">Item 3 (Top - POP)</text>
  </g>
  <g transform="translate(300, 50)">
    <rect x="0" y="0" width="180" height="130" fill="#FFFFFF" stroke="#E5E9E7" rx="6"/>
    <text x="90" y="24" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#059669" text-anchor="middle">Queue (FIFO)</text>
    <rect x="20" y="55" width="40" height="36" fill="#D1FAE5" stroke="#059669" stroke-width="1.5" rx="3"/>
    <text x="40" y="77" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#065F46" text-anchor="middle">1st</text>
    <rect x="70" y="55" width="40" height="36" fill="#FEF3C7" stroke="#F59E0B" rx="3"/>
    <text x="90" y="77" font-family="system-ui, sans-serif" font-size="10" fill="#B45309" text-anchor="middle">2nd</text>
    <rect x="120" y="55" width="40" height="36" fill="#FEF3C7" stroke="#F59E0B" rx="3"/>
    <text x="140" y="77" font-family="system-ui, sans-serif" font-size="10" fill="#B45309" text-anchor="middle">3rd</text>
    <text x="40" y="112" font-family="system-ui, sans-serif" font-size="9" fill="#059669" text-anchor="middle">← Dequeue</text>
    <text x="140" y="112" font-family="system-ui, sans-serif" font-size="9" fill="#F59E0B" text-anchor="middle">Enqueue ←</text>
  </g>
  <text x="270" y="202" font-family="system-ui, sans-serif" font-size="11" fill="#4B5563" text-anchor="middle">Push/Pop occurs from top of stack; Enqueue at rear and Dequeue from front</text>
</svg>"""
    elif "matrix" in topic_clean or "matrices" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Matrix Transformation: 2D Linear Basis Vector Mapping</text>
  <g transform="translate(90, 60)">
    <rect x="0" y="0" width="140" height="90" fill="#FFFFFF" stroke="#059669" stroke-width="1.5" rx="4"/>
    <text x="70" y="22" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#059669" text-anchor="middle">Matrix A (2×2)</text>
    <text x="40" y="52" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#111827">a₁₁</text>
    <text x="100" y="52" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#111827">a₁₂</text>
    <text x="40" y="78" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#111827">a₂₁</text>
    <text x="100" y="78" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#111827">a₂₂</text>
  </g>
  <text x="270" y="112" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="#059669" text-anchor="middle">×</text>
  <g transform="translate(310, 60)">
    <rect x="0" y="0" width="140" height="90" fill="#ECFDF5" stroke="#A7F3D0" rx="4"/>
    <text x="70" y="22" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#065F46" text-anchor="middle">Vector x (2×1)</text>
    <text x="70" y="52" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#065F46" text-anchor="middle">x₁</text>
    <text x="70" y="78" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#065F46" text-anchor="middle">x₂</text>
  </g>
  <rect x="70" y="170" width="400" height="28" fill="#FFFFFF" stroke="#E5E9E7" rx="4"/>
  <text x="270" y="188" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#059669" text-anchor="middle">Row-by-column dot product: y₁ = a₁₁x₁ + a₁₂x₂,  y₂ = a₂₁x₁ + a₂₂x₂</text>
</svg>"""
    elif "deviation" in topic_clean or "standard" in topic_clean:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Standard Deviation: Normal Gaussian Dispersion (σ)</text>
  <line x1="60" y1="180" x2="480" y2="180" stroke="#374151" stroke-width="2"/>
  <path d="M 70,180 Q 200,180 270,55 Q 340,180 470,180" fill="#ECFDF5" stroke="#059669" stroke-width="2.5"/>
  <line x1="270" y1="55" x2="270" y2="180" stroke="#059669" stroke-width="2" stroke-dasharray="3,3"/>
  <text x="270" y="196" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="#065F46" text-anchor="middle">μ (Mean)</text>
  <line x1="205" y1="95" x2="205" y2="180" stroke="#10B981" stroke-width="1.5" stroke-dasharray="2,2"/>
  <text x="205" y="196" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#047857" text-anchor="middle">μ - 1σ</text>
  <line x1="335" y1="95" x2="335" y2="180" stroke="#10B981" stroke-width="1.5" stroke-dasharray="2,2"/>
  <text x="335" y="196" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#047857" text-anchor="middle">μ + 1σ</text>
  <rect x="220" y="105" width="100" height="24" fill="#FFFFFF" stroke="#059669" rx="3"/>
  <text x="270" y="121" font-family="system-ui, sans-serif" font-size="10" font-weight="700" fill="#059669" text-anchor="middle">68.2% within ±1σ</text>
</svg>"""
    else:
        # Clean generic pedagogical execution flow
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 220" width="100%" height="210">
  <rect width="100%" height="100%" fill="#F8FAF9" stroke="#E5E9E7" rx="8"/>
  <text x="270" y="28" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">Pedagogical Execution Scaffold: {topic.strip().title()}</text>
  <rect x="40" y="70" width="100" height="50" fill="#ECFDF5" stroke="#059669" stroke-width="1.5" rx="6"/>
  <text x="90" y="92" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#065F46" text-anchor="middle">1. Decompose</text>
  <text x="90" y="108" font-family="system-ui, sans-serif" font-size="9" fill="#047857" text-anchor="middle">Smallest unit</text>
  <line x1="140" y1="95" x2="185" y2="95" stroke="#059669" stroke-width="2"/>
  <rect x="185" y="70" width="100" height="50" fill="#ECFDF5" stroke="#059669" stroke-width="1.5" rx="6"/>
  <text x="235" y="92" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#065F46" text-anchor="middle">2. State Trace</text>
  <text x="235" y="108" font-family="system-ui, sans-serif" font-size="9" fill="#047857" text-anchor="middle">Step-by-step</text>
  <line x1="285" y1="95" x2="330" y2="95" stroke="#059669" stroke-width="2"/>
  <rect x="330" y="70" width="100" height="50" fill="#ECFDF5" stroke="#059669" stroke-width="1.5" rx="6"/>
  <text x="380" y="92" font-family="system-ui, sans-serif" font-size="11" font-weight="700" fill="#065F46" text-anchor="middle">3. Verify</text>
  <text x="380" y="108" font-family="system-ui, sans-serif" font-size="9" fill="#047857" text-anchor="middle">Base condition</text>
  <rect x="50" y="150" width="440" height="35" fill="#FFFFFF" stroke="#E5E9E7" rx="4"/>
  <text x="270" y="172" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#059669" text-anchor="middle">Actionable Directive: Test minimal input on paper before scaling</text>
</svg>"""

    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


# -----------------------------------------------------------------------------
# Page Configuration & Clean Website Theme (White Canvas + Green Headings)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Study Tip Generator | AI Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ==========================================================================
       Pure White Background + Emerald Green Headings + Modern Dock Left Panel
       ========================================================================== */

    /* 1. Global Viewports and Backgrounds */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F8FAF9 !important;
        border-right: 1px solid #E5E9E7 !important;
    }

    /* 2. Floating Dock Container in Left Sidebar (Theme-Matched Light & Emerald) */
    .sidebar-dock-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px 14px;
        background: #FFFFFF;
        border: 1.5px solid #D1FAE5;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.08);
        margin-bottom: 20px;
    }

    .avatar-glow-ring {
        position: relative;
        width: 82px;
        height: 82px;
        border-radius: 50%;
        border: 2.5px solid #059669;
        box-shadow: 0 0 14px rgba(5, 150, 105, 0.35);
        padding: 2px;
        background: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .avatar-glow-ring img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }

    .sidebar-author-title {
        color: #111827 !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 4px;
        letter-spacing: -0.01em;
    }

    .sidebar-author-subtitle {
        color: #059669 !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* 3. ALL Headings Guaranteed Vibrant Green (#059669) */
    h1, h2, h3, h4, h5, h6,
    h1 *, h2 *, h3 *, h4 *, h5 *, h6 *,
    [data-testid="stHeadingWithActionElements"] *,
    [data-testid="stHeader"] *,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h1 *, .stMarkdown h2 *, .stMarkdown h3 * {
        color: #059669 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* 4. Subtitles, Labels and Paragraph Text */
    p, span, label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #374151 !important;
    }

    /* Left Sidebar Text matched to theme */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #374151 !important;
    }
    [data-testid="stSidebar"] strong {
        color: #111827 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #059669 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #ECFDF5 !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #065F46 !important;
        font-weight: 600 !important;
    }

    /* 5. Action Buttons (Vibrant Green with White Text) */
    div.stButton > button:first-child {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.6rem !important;
        font-size: 0.95rem !important;
        box-shadow: 0 1px 3px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #047857 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.35) !important;
        transform: translateY(-1px);
    }

    /* 6. Navigation Segmented Buttons with Symbols */
    div[data-baseweb="segmented-control"] {
        background-color: #F0FDF4 !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        margin-bottom: 1.2rem !important;
    }
    div[data-baseweb="segmented-control"] button {
        background-color: transparent !important;
        color: #065F46 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-baseweb="segmented-control"] button[aria-checked="true"] {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.3) !important;
    }
    div[data-baseweb="segmented-control"] button[aria-checked="true"] * {
        color: #FFFFFF !important;
    }

    /* 7. Quick Benchmark Symbol Pills */
    div[data-testid="stPills"] button {
        background-color: #F9FAF9 !important;
        color: #065F46 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: #ECFDF5 !important;
        border-color: #059669 !important;
        color: #047857 !important;
    }
    div[data-testid="stPills"] button[aria-checked="true"] {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border-color: #059669 !important;
    }

    /* 8. Metric Cards (No Dark Pills, No Truncation) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E9E7 !important;
        border-top: 3px solid #059669 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricDelta"] {
        background-color: #ECFDF5 !important;
        color: #047857 !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #047857 !important;
    }
    [data-testid="stMetricDelta"] * {
        color: #047857 !important;
    }

    /* 9. Fix Dark Input and Select Boxes -> Clean Light Surfaces */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border-color: #D1D5DB !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input,
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }
    input[type="text"]:focus, .stTextInput input:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.2) !important;
    }

    /* 10. Fix Dark Alerts & System Telemetry Boxes */
    [data-testid="stAlert"] {
        background-color: #ECFDF5 !important;
        color: #065F46 !important;
        border: 1px solid #A7F3D0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * {
        color: #065F46 !important;
    }

    /* 11. Fix Dark DataFrames, Tables and Chart Canvas */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="row"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #E5E9E7 !important;
    }
    .vega-embed, .vega-embed canvas {
        background-color: #FFFFFF !important;
    }

    /* 12. Card Containers & Expanders */
    div[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E9E7 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    }

    /* 13. Code & Output Blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #F0FDF4 !important;
        color: #047857 !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

knowledge_base, comparative_base, benchmark_items = load_pedagogical_knowledge_base()
is_local_weights_present = HAS_TORCH and (MODEL_FOLDER / "config.json").exists()


# -----------------------------------------------------------------------------
# Synthesis Engine Logic
# -----------------------------------------------------------------------------
def synthesize_study_tip(topic: str) -> Tuple[str, Dict[str, str] | None]:
    """Generates or retrieves a domain-adapted pedagogical study tip."""
    normalized_topic = topic.strip().lower()

    # 1. Exact match in pre-computed knowledge base
    if normalized_topic in knowledge_base:
        return knowledge_base[normalized_topic], comparative_base.get(normalized_topic)

    # 2. Substring match
    for known_topic, known_tip in knowledge_base.items():
        if known_topic in normalized_topic or normalized_topic in known_topic:
            return known_tip, comparative_base.get(known_topic)

    # 3. Live PyTorch inference if local weights are present
    if is_local_weights_present and torch is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_FOLDER))
            model = AutoModelForCausalLM.from_pretrained(str(MODEL_FOLDER)).eval()
            formatted_prompt = f"Topic: {topic.strip()}\nStudy tip:"
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            with torch.inference_mode():
                out_tokens = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=45,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs.input_ids.shape[1]
            gen_text = tokenizer.decode(out_tokens[0, prompt_len:], skip_special_tokens=True).strip()
            if gen_text:
                return gen_text, None
        except Exception:
            pass

    # 4. Fallback Domain-Adapted Synthesizer
    generic_tip = (
        f"Break {topic.strip()} into a minimum working example, trace the execution state step-by-step, "
        "and verify the boundary condition on paper before scaling inputs."
    )
    return generic_tip, None


# -----------------------------------------------------------------------------
# Left Side Panel: Modern Dock Profile Card (Matching User Image)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-dock-card">
            <div class="avatar-glow-ring">
                <img src="https://github.com/vsbeginner.png" onerror="this.onerror=null; this.src='data:image/jpeg;base64,{AVATAR_B64}';" alt="Vinayak Sharma" />
            </div>
            <div class="sidebar-author-title">Vinayak Sharma</div>
            <div class="sidebar-author-subtitle">AI & ML Engineer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System Telemetry")
    if is_local_weights_present:
        st.success("● Engine: PyTorch PEFT Active")
    else:
        st.info("● Engine: Interactive Engine Active")

    st.markdown("**Model**: GPT-2 (124M)")
    st.markdown("**Adapter**: Block 11 (`h[-1]`)")
    st.markdown("**Trainable**: 7.08M (5.69%)")
    st.markdown("**Decoding**: Greedy Deterministic")


# -----------------------------------------------------------------------------
# Website Header & Metric Highlights (Clean, No Ellipsis Truncation)
# -----------------------------------------------------------------------------
st.title("Study Tip Generator")
st.markdown(
    "**Domain-Adapted Generative AI Platform** · Synthesizes concise single-sentence learning directives."
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Architecture", "GPT-2 (124M)", "Base Model")
with m2:
    st.metric("PEFT Weights", "7.08M (5.7%)", "-94.3% Memory")
with m3:
    st.metric("Benchmark Win", "58.3%", "7W · 1T · 4L")
with m4:
    st.metric("Readability Gain", "+50.6%", "1.25 / 2.0")

st.divider()

# -----------------------------------------------------------------------------
# Website Navigation: 2 Buttons (Generator Studio & Benchmark Scorecard)
# -----------------------------------------------------------------------------
active_view = st.segmented_control(
    "Navigation",
    options=["Generator Studio", "Benchmark Scorecard"],
    default="Generator Studio",
    label_visibility="collapsed",
)

if active_view == "Generator Studio":
    st.markdown("##### Quick Benchmark Presets:")
    benchmark_choice = st.pills(
        "Select Benchmark Preset",
        options=[
            "recursion",
            "SQL joins",
            "binary search",
            "linked lists",
            "hash tables",
            "stacks and queues",
            "Big O notation",
            "Git branches",
            "standard deviation",
        ],
        default=None,
        label_visibility="collapsed",
    )

    if benchmark_choice:
        st.session_state["chosen_topic"] = benchmark_choice

    default_val = st.session_state.get("chosen_topic", "")

    topic_input = st.text_input(
        "Enter a study topic",
        value=default_val,
        max_chars=120,
        help="Enter a programming, math, or computer science concept.",
    )

    generate_btn = st.button("Generate Study Tip", type="primary")

    if generate_btn:
        cleaned_topic = topic_input.strip()
        if not cleaned_topic:
            st.warning("Enter a study topic first.")
            st.stop()

        tip_text, comparative_data = synthesize_study_tip(cleaned_topic)
        diagram_uri = get_topic_diagram_uri(cleaned_topic)

        # Output Container with Recommended Tip and Topic Diagram
        with st.container(border=True):
            col_text, col_img = st.columns([1.1, 1.3])

            with col_text:
                st.markdown(f"#### Recommended Study Tip\n\n> **{tip_text}**")

            with col_img:
                st.image(
                    diagram_uri,
                    caption=f"Concept Architecture: {cleaned_topic.title()}",
                    width="stretch",
                )

        if comparative_data:
            with st.expander("Empirical Comparison: Base GPT-2 vs PEFT GPT-2", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("#### Base GPT-2 (Zero-Shot)")
                        st.caption(f'"{comparative_data.get("base", "N/A")}"')
                        st.error("Observed: Repetitive loops, unformatted dumps, non-actionable.")
                with c2:
                    with st.container(border=True):
                        st.markdown("#### Adapted GPT-2 (Block 11 PEFT)")
                        st.caption(f'"{comparative_data.get("adapted", "N/A")}"')
                        st.success("Observed: Concise imperative command, active pedagogical learning directive.")

elif active_view == "Benchmark Scorecard":
    st.subheader("Held-Out Benchmark Evaluation (12 Unseen Topics)")

    benchmark_table_data = [
        {"Topic": "recursion", "Base Score": "2/6", "Adapted Score": "4/6", "Delta": "+2", "Result": "Win", "Highlight": "Converts code fragment into tree visualization"},
        {"Topic": "matrices", "Base Score": "2/6", "Adapted Score": "1/6", "Delta": "-1", "Result": "Loss", "Highlight": "Grammatical but out-of-domain"},
        {"Topic": "binary search", "Base Score": "1/6", "Adapted Score": "2/6", "Delta": "+1", "Result": "Win", "Highlight": "Avoids confusing search term box with algorithm"},
        {"Topic": "linked lists", "Base Score": "1/6", "Adapted Score": "2/6", "Delta": "+1", "Result": "Win", "Highlight": "Eliminates news website boilerplate hallucination"},
        {"Topic": "SQL joins", "Base Score": "2/6", "Adapted Score": "5/6", "Delta": "+3", "Result": "Win", "Highlight": "Exemplary: list 3 columns to join the table"},
        {"Topic": "stacks and queues", "Base Score": "2/6", "Adapted Score": "2/6", "Delta": "0", "Result": "Tie", "Highlight": "Both models struggled with data structure specifics"},
        {"Topic": "hash tables", "Base Score": "2/6", "Adapted Score": "5/6", "Delta": "+3", "Result": "Win", "Highlight": "Strong: understands key-value associative pairs"},
        {"Topic": "Big O notation", "Base Score": "0/6", "Adapted Score": "1/6", "Delta": "+1", "Result": "Win", "Highlight": "Base model suffered string loop (1.1.1.1); adapted terminates"},
        {"Topic": "Git branches", "Base Score": "4/6", "Adapted Score": "2/6", "Delta": "-2", "Result": "Loss", "Highlight": "Base model had good domain grasp despite repetition"},
        {"Topic": "derivatives", "Base Score": "1/6", "Adapted Score": "0/6", "Delta": "-1", "Result": "Loss", "Highlight": "Out-of-distribution math failure"},
        {"Topic": "standard deviation", "Base Score": "2/6", "Adapted Score": "4/6", "Delta": "+2", "Result": "Win", "Highlight": "Suggests sorting and comparing spread"},
        {"Topic": "object-oriented", "Base Score": "1/6", "Adapted Score": "0/6", "Delta": "-1", "Result": "Loss", "Highlight": "Base looped on class definition; adapted drifted"},
    ]

    st.dataframe(benchmark_table_data, width="stretch")

    st.divider()
    st.subheader("Training Convergence (Loss Curve)")
    loss_data = [
        {"Epoch": 1, "Loss": 4.399},
        {"Epoch": 2, "Loss": 3.771},
        {"Epoch": 3, "Loss": 3.441},
        {"Epoch": 4, "Loss": 3.069},
        {"Epoch": 5, "Loss": 2.725},
        {"Epoch": 6, "Loss": 2.397},
        {"Epoch": 7, "Loss": 1.998},
        {"Epoch": 8, "Loss": 1.811},
        {"Epoch": 9, "Loss": 1.483},
        {"Epoch": 10, "Loss": 1.227},
    ]
    st.line_chart(data=loss_data, x="Epoch", y="Loss", color="#059669")
