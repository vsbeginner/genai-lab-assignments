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
    "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"
    "60oOmeI20B11u7/e///2U+6Fz1U5Xy0+fPE3r143E51uP/Z/8x1/Rz//rcbZ2kM6EskA5wZq+973"
    "voY9+9KOs2A5s20rvuPMu5unE6/aBwXvuvpvuuusuVnBf+eY36f9++Uv29pD09/bQgw/8U8tVhh7"
    "0z/6vP6P77ruP173+5p/8CX1v+jT/hge98y1vonvuvYfnNHz/hz+ix/7reZb6gQf+qW10F8qBwP7"
    "eG26i//qv/4oV7b/8y79kLxFv/D3f+R80PDxMb7jhBhocGKRbbn4Tvfvd72bg/+eXP6dfP/kkhwG"
    "5d7g8Sve9/Q5629vehcWk2w7eTj/61z+zt3j4yBH68z/+U2wN/+d//gc9duo0PfzQg/T73z/BnjH"
    "m6vve/wF60xvewM976aWX6Nvf/jb/Dcf/6z//k57/3e+Y55D96u5u+vjHP0Ef+tD7eZlD4Hn0kUf"
    "p108+Qddfe4CGBvroqquuooceei9bM3/9y5/Ts889Ry+99BLf9+FHHqH+vh66/T3voze/6abY/gY"
    "4/u4Pf0AvHTlCG9avp5/97Gdcr5/85Cd87L777uPnPe/972eFv+766+iN12+n66+/gW64/nqO+vj"
    "FL/6brWd0X4Tf/4eHHybPdWjLxg30rne9m22D97/nfbRu7Thdf/31tG/vHnrTm95Er9t+jZnWc9y"
    "48bY73ki7dtxABw/uowP799L6teuYx/fsvo55+7abD9CN119L27dcRVetHqHBwUH2nPFvG7Zvpbv"
    "vvpu52+jW//z85/SLX/wPvfjii/w8rB+/fOIJ+vWvf81Xf+9734tXQe3bdx1t3bKR/W70DNDeW95"
    "0E123fTtt2rSBtm/fSmOjoxwd+c9//Uf67W9+w16Z1pNPPE7/9Z//yb/vf//7Y/c7sH8/bdu2jW7"
    "edT3t2LqVdu7cyddt2LiRxkYGaGhkjG648SZ6/fV7mG+hG5iB/V2hXbv20eFDh2h4uI8G+ntp25a"
    "tNDY6SgN9fbRzxw665aab2f4/uG8/rd+wgVavHqEB2M2Dgwzg+s3r6R1vfRMt1au0du047d1zPXs"
    "r4Ljh/j66bu/1dNMNN7FihXKB8n/f/ffR6NCAuX/wD39gW/6vfvkr/t7vf/97/g7e0cMPP0w//elP"
    "aXFxkW0d/H7l5ZfpxRdfjMfBfv/LX/D3f/ub37DnjjJ+6YUXaH5+ni6cP0+PP/44XbhwgS5evMij"
    "E4hH/8ff/y1duHCBXj7xMj36H4/S+QsX6Py5C/Tcc8/T/MIinTlzhn795JMsZfNzF+ipJ5+mc+cu"
    "0Osvv0QvHz1KZ86epcsXL9LjTzxJl+cW6cL5eTpz5hwdP3GSpqam6Nz5C/T000/TlcuX6eLFCfp/"
    "z07Q5QsX6aWXXqLLly/Tlfn52Pq1qG9+fp7Onj1LZ8+eZa7G9fPzF+nM2TNs7SBo//nnn2dg/vKX"
    "v8y9C1x3bOIYvfnNb+Zp2d///vdsfSBo/9SnPsX/g1WAP3h1+Zkzp+jJJ59kAP/617+ykv/ud79j"
    "qwL3vXLlCr366qsM7KuvvsrXDQwMsHf8wx/+kC1L/A1hUHg8wN9111306U9/mkGdn59n3v3yl7/M"
    "HgG8N5QZ3q7v+7R7924G+m9/+xsDjeuhzHDv8+fP814F7n311VfT0tISfepTn4pdG2W44447mNuR"
    "vwsXLvC45/nz59lae/nll9mzefnll3kfB8r0s5/9LNsvuO4Xv/gFAw06wh5/3eteF3u9KBPAgV4x"
    "1Y4vD6U6OTkZ8zn61ZtvvpndC+hN718vLS3xNOHixYs8rQJtPPLII/SrX/2Kn1Wv1+m1117jQf8L"
    "Fy7EkwVw97e85S1s34P3AejPfvaz9Pd///f8/G9/+9s8oXDhwoXYi8H1P//5z/m/f/jDH2K7G77n"
    "P/7jP9gnxsCFeC78hveN3tDr16+Pd4Hwzvvf/362B44dO8Y9CejN1772Nf4+eheAgT711FP0ne98"
    "h8+h4eCZo/fwf//3f7l38Jvf/Ib92GeeeYbB068L4C88A3g4sAMj7+LFCf7/1FNPxZ4U9k54Y7y3"
    "h4X67rvv5v0a8HbgqfT19bGv861vfYsnw6A4uB/v6+Aee+yxv/74x//m7b86gD//+c9jBwE0g/0I"
    "w8Xf3/nOd/h53/72t1kJoT/06KO/ZXsLPA9oG/u98B7w/kCnoK2f/OQnvA8EOgY4gAMwAGBwcJDz"
    "YtG9eXh3aAh961vfZM8fvXmAHr9133vttdfs37t3b95O6A996EP0ox/9iO2V+s4776Qf/ehHbK8B"
    "tI/88If0/ve/n/3Qd7zj7XT/ffezP/bBf/wAvfvuu+h973sf/esDD7BN8cEPfoAevP9e9qj/6bZ3"
    "03333Ud33HEH/dOd76T9+3bTf/7nZzmA/P53v5u55+6776Fbb72Vbtt/E91+3U66YccWuuuuu5ja"
    "4d/r+e1vfx/9x3/8B739bW9j8IeGBtiuB0B2bN/G195z9z2s+G65+Tq6/tob6IYbbuC8uLdu+1ba"
    "s+Mquuqqq3jD829vupGu230tvflNb6LDhw/zPgd6K3s2bGDA0d258847OQoO1w687W30/vd/gH70"
    "ox/x/fD3gT/4AO3b9zr6wAc+QAcPvk6e/95777W/941v0N59+2nf/mvp7rvvYZ4E19266zq65Zab"
    "aP/r9tA111xD1113HX+OewPwbdu28e+73/1uuuqqq/h/KAUoT3B972+8jtauXUuvvfYaWrVqFfeU"
    "rrrqKtqwYQNbN0g0rB3g2967F3mdfu/7Xkdbduykm193F121eg1tXL+BNr12I23bsYO2bN1MV+28"
    "ltatXUe/853vsCL8f+/7H3T9bT+gfbt30vU3v45uuPlmft2rr76aVq9ezdYA1q1fv57/h9K76aab"
    "uEfh/950003cbV111VVs3V/12mtp5cqV9MY3vpG5/qqrrqIrV67kxwM7lB1eBfrwhz/MlYQ9DRB3"
    "7tzJE1Bbt27lkMqaNWtY+a1bt44dGNABaPCOHTv4dytWrKDrr7+elSt6e1u3bh3vAa5bt453YmDr"
    "bt68mXfG8Dr3/m9/k/bu3Ea37N5O+/fsot2v200bNm6gLVevoc1XvYZ2XLWdNm7cSOtGBmnr5k30"
    "R3/0R/z3f33sMXrt1TfQe++8je69/SZav/kqumH/jVwx2IdA14cPH6bf//73/N76y1/+Mm3cuJF2"
    "797NVgC4sXbtWn6/hN9X/f2aNWy5YI+B3rBl8xZad/Vo9H5b34s3bdrEew1gq48fP04//OEPef+F"
    "3h7o/e3atYs+8YlP8N9bb72VrcFnn32WwUaZ3v72t/P7a9jL3HHHHeydwh5j//79fO8bbriB97tA"
    "V7g3+u63v/1tngeP8731rW9lv89973vfO9f/xYF+/dveRjffejO9/rrtdM01m2jH9uu58qA4gA5B"
    "e27YtJ62bd/MV/78V7/k/195/lU6fW6C5hd61D8wSDffejtt2baV58kBsN27d/P/Tz31FP3mN7/h"
    "925w/02bNvE9du7cSVdeeaUd6b3qKjp48CB7e/C0fvazn9HLL7/Mv4eGhujqq69m1wW7/bprtvA+"
    "bN/+G9kLOH/+PA+l/fSnP6Wnn36av5c/D/fceustXN/3vOc99KEPfYivxfs3eA56H1D6X/ziF2yv"
    "ve9976OPfOQjPI4I2gfdYv+FsgPdwK7H9c8++yw/Z9++ffTZz36W3w9985vfZO/gX778RfphX1/i"
    "d6uH6d/f/Vf0r2+9na7evJHWb7mKNmzbQv29XXTtddfSO959N33uC39L5y9M8g4N0D/5zK/pT37v"
    "9+kdb387jY4O0+joKPc4XnnlFXrppZf4d8eOHXz86quv5vfR1q5dy7sC27ZtY+8R7+fhnqeffpoV"
    "0fXr1/O+061veQt97GMf43fVtm3b+H5/9atf0c///M/p7NmzPH9+/fp19M1v/i198pOfZHscdgR8"
    "P/y/ePEi/elPf4p/v/e979F73/te/h3nQfmg/1tvvZXfL2DvAeyX8P+f/OQnvAdBH4x32+C5oPzP"
    "Pfccj1kCPLhv3z6u85//9m/p13190d+N6+m+f/53uvud76KNV62j1WvW0KqRoTgqfOdt9LHPfIGe"
    "PfYiL5v/5W9+R3/6qY/SLW+6mb7/r0/QM8+/wKMLN954I79HffHFF+kvf/kLvwfG1atXj9CmTZt4"
    "wwhf/vKXv8z7NnfffTffFwGvd7/73eytAHz07fBeA/7g3Qfc/+abb+ZXA93f8Y538L99fX18/1/8"
    "4hf06KOPclR53bp1dPDGG+ktb30LfepTnxpH7Lvfzf7iBz7wAdYh/vCHP8y9C1x/9dVXcwWuu+46"
    "1gmAF0A/99xzrNzveMc7eK8B7o83xH/18CP0r+vWRW+Yd9MffuB9tP91u2mkv4e2bd1MV60eoX/8"
    "yN/Rk8++yAP8b3/nefrol/6afvu2vTz0t2XzBvr3B/6Jvv3I99kyWLduHW3bti2ODr70En/7y1/+"
    "kr9funQpHlV405veROvXr+edBOxBoGegkIeGhthGfvWrX00PPvggb+SjX3XPPffw/hG+v/fee/l9"
    "9HOf+1y8yQL4hUq95JJL2B7Eawf4ffXVV9O3vvUt7i3gffT69evZvwf7Hkrrk5/8JCvyT33qU2wt"
    "Pv3001w/AP7Zz36WvU4oO5wH3f/p4UfoZ+vXJ+85jLz9r3T1VVfS6lUjtGndOq6gH4S0adNG+trf"
    "/p5eefU1+vmvf80fI5/97R/Q//7zP9KNt1xDn//Cl+ir//RN3k/A8euvv57279/Pv1944QX+ffHF"
    "F/n9EOzP9evXc6H27NnD9z948CC99a1v5dfXoQv8Xve619HOnTsZ9Icffpi9pWuuuYbv/aEPfYg+"
    "/elPs4eI6yBNsBeBHoGyA1h47/vud7/LSwdAGU6dOpU6uHh80003cbVhpWDfAOw88E7/6le/4t8f"
    "/vCH4wH+/ffzZ3Ff9GZw/1tuuYUuXrxIz7/wAn3r6tXRe7m6unrYy19/7Q7a8eY9PLNq9+5r6fXX"
    "bqa9126nH/zLv9PZCwv09LPP07e+9yP657u/QLf/9r/T79x+K33jG9+iDz3wQfbWtm7dyoXbu3cv"
    "P+/Xv/41/4aEBh2uXr26BfTBgweZG8DOoId+5pln6Omnn6Y//elPDPTtt9/O+2sPPfQQvfOd72Rg"
    "T548yfY1egY4FwoJ5QWbEN89ePAg9+TwnoK+HwCE8yGtsFwAAt7Bww8/zO8v7rjjDnrfe99L1+ze"
    "wfa6Vq1ahc71hQsX6Pnnn6fnjhylr284EHv3vT399K//8gDdevPNdOO119BVq1exk9PTU6a+3m5a"
    "OTxEBw/eTj969Fn67988yZ7aE39+hn73X/6F3nPnzfTHn/4y/c39f0Pf/e53aPv27bRu3brU06H3"
    "7t3Lv8+cOcO8jHjMpk2beA8Bv9etW8cK58EHH6RbbrmF51H8+te/pvvvv5/fB8De8L4b3p3Dnvk7"
    "3/lO7C2hvG9729u4h4T3wL//+7/PXhb46h3veAffC0AC4Ntvv517Rjff/Hq65aab2H19z3vew3sJ"
    "eD9w1/U30KZN19CePdexF/i3zz5LPztwIAa+q6eH3vf/fIj27b+Btm7ZRH3lMuUKBVY6hVKJymWb"
    "0v/6l0/QM8+/yJ4mZlV+9V9+yF7+X//lX9AHP3g/e8Zf//rX2f5FpYI2g13x1FNPsd8Kz8KkAO9L"
    "r169mnsE3/nOd3g3D+8z4P7vfe97ea/p85//PPuL8H5w/b//+7/n33j/C76yZ02+8pWvxK+6w3k4"
    "j8G6kHaAFp0f16L8N954I482YFz03e9+N3sfqMhvufl1dPvtt/P7k5+fOEm/eOtbY9l0u7v66b1v"
    "ezvdfssttH7tOhZk7vXnB9TdVWJ9s2XzBrp6zTC9871vp0f+61k6ff4ifec7P6RP/uVn2ev6u099"
    "knbv3k0bNmxIBbrf/OY36Uc/+hE98cQTDDT4g142PDJ4gXfeeSfvZ+D9B9z7k5/8JCvy3bt3c68C"
    "gIOWsK8B/r7zne9kex69N3jOq1at4u/hnCeffJKH0uCZw8vEOYyS4/sA6wO/4T64B/R861vfSq+7"
    "fjf91m/dxqBjdYI+B2VdObycfvWJTyWq43D3v/sdb6N3ve0WuuWmW+iGG26m9euvoZUDfdRT7uYe"
    "eP1YHz0/Xafv/ug3tFhfoP/90m/oB9/9Pt3+1r+md739bXTPu95NDz/8MHtz0Wgc/I8//3l66aWX"
    "2I8Ffwf3Q5Lg8X/rW99ir/q9730v70fAvfCOHfaX3v72t/PA/+c//3l+X4FvN2/ezMDBI8T9UdbP"
    "fOYz/J+494c//GEGBG8T4PnAGFSA3hDug6WzdevW8bvj22+7nW6++Wa68bprWbHddNNb4hF2161f"
    "f2B/b18f3XTTTfTud72bbnrtG9gG7+8pU0+5m/XJ/n376J6730F3v+s2uv32W+iOW2+h63dtoxsu"
    "U4+f/exnvE+B97fgI/fss8/y+x54B/g7PDZ469jHg/fG+/n777+fexZ33nknj1Pg7xNPPBFLI+gW"
    "dABe4nve8x4eP0Rv8hOf+AQvW/ztb3/L98J7EHgq4F8ABZ4Nrgf/4bve9W5673vfS+973/v4dfgZ"
    "b8t/5VvfGivxO+/Yv6+v9y30kQ9/hK5avY6GBvqprydP+f6AiqUuGhnup+3bt9OOHdvp+m3X8P4F"
    "9pS2X72Vdu64iq7bdi3t3rmTNm/eTN3d3Sy1qKTA+r333sveG97Twj1Q8j/5yU/oySef5PFK3Iff"
    "f/CDH+R77N27l4GG1wL+8qA8/C7mZ+C8eA6U63e/+12+B77D+/747kY3wN9Qclxvf+Mb38g8j16f"
    "mI69e/fylI14/u985zv0z//8z/y1eD5eFfSvv+991F/uodtuuYUG+rrJL3VTT3mQypVC/F354N8D"
    "A33RfyF1l0sp6I8XGq9f37Nnj7l4Vd3XfN8Y2w7s0j2kXoUet9F+V4eU1X016489P+2s+8m9s+zK"
    "601zX9u/T+7r97f+vK5fX03f757/B930/jX7Hw0cAAAAAElFTkSuQmCC"
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
                <img src="data:image/png;base64,{AVATAR_B64}" alt="Vinayak Sharma" />
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
