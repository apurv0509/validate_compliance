#!/usr/bin/env python3
import json
import re
import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from typing import List, Optional
import uuid

# NEW import for the Pythonic docling conversion
from docling.document_converter import DocumentConverter

app = FastAPI(title="Compliance Check API")

# Configuration
class Config:
    CACHE_DIR = Path("./.cache")
    MAX_CHUNK_SIZE = 500
    POLICY_MODEL = "gemini-2.0-flash@vertex-ai"
    COMPLEX_MODEL = "claude-3.5-sonnet@anthropic"
    OPENAI_BASE_URL = "https://api.unify.ai/v0/"
    OPENAI_API_KEY = os.getenv("UNIFY_API_KEY", "lObMF7AmHZkDJzZtV30o3Ncy+P60uv+iELR8pY6cZlY=")

class PolicyRequest(BaseModel):
    policy_url: str
    content_url: str

class Violation(BaseModel):
    policy_phrase: str
    violation_phrase: str
    explanation: str

# Initialize clients
sync_client = OpenAI(
    base_url=Config.OPENAI_BASE_URL,
    api_key=Config.OPENAI_API_KEY
)

async_client = AsyncOpenAI(
    base_url=Config.OPENAI_BASE_URL,
    api_key=Config.OPENAI_API_KEY
)

# NEW: Replace the previous CLI-based run_docling with this pythonic version.
async def run_docling(url: str, prefix: str) -> Path:
    """
    Use the Pythonic DocumentConverter from the docling package to convert a given URL
    (or local file path) into markdown. The resulting markdown is written to a file in CACHE_DIR.
    """
    Config.CACHE_DIR.mkdir(exist_ok=True)
    filename = f"{prefix}-{uuid.uuid4()}.md"
    output_path = Config.CACHE_DIR / filename

    def convert_and_save():
        converter = DocumentConverter()
        result = converter.convert(url)
        markdown = result.document.export_to_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    # Run synchronous conversion in a thread
    await asyncio.to_thread(convert_and_save)

    if not output_path.exists():
        raise HTTPException(500, detail="docling conversion failed to create output file")
    return output_path

# (The remainder of the code remains unchanged)
class PolicyAnalysis(BaseModel):
    terms_to_avoid: List[str] = []
    recommended_terms: List[str] = []
    complex_rules: List[str] = []

class PolicyAnalyzer:
    def __init__(self, openai_client: OpenAI, model: str = Config.POLICY_MODEL):
        self.client = openai_client
        self.model = model

    def _preprocess_policy(self, policy_text: str) -> str:
        lines = [
            line.lower()
            for line in policy_text.split("\n")
            if "🖼️❌ image not available" not in line.lower()
        ]
        return "\n".join(lines)

    def _extract_section(self, policy_text: str, section_title: str) -> (str, str):
        section_pattern = rf"##\s+{re.escape(section_title.lower())}\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(section_pattern, policy_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            start, end = match.span()
            remaining_text = policy_text[:start] + policy_text[end:]
            return (extracted, remaining_text)
        else:
            return ("", policy_text)

    def _analyze_with_openai(self, policy_section: str) -> PolicyAnalysis:
        prompt = (
            f'For the below compliance policy document (markdown text), given some content I want to check if there are any violations:\n'
            f'For this task I want the policy to be parsed into "terms_to_avoid" and "recommended_terms". '
            f'Since I will run regex on the content using your output, ensure that your output is regex appropriate.\n'
            f'Some of the policy rules will be more complicated such that direct regex will not work. '
            f'Take these complicated policy rules and give them to me as a list of "complex_rules". '
            f'Each policy rule should be present in one of these three.\n'
            f'Output a JSON containing these three keys. Be thorough.\n\n'
            f'POLICY:\n'
            f"```\n{policy_section}\n```"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        try:
            resp_content = response.choices[0].message.content
            json_response = json.loads(resp_content)
            return PolicyAnalysis(**json_response)
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError("Failed to parse OpenAI response") from error

    def analyze(self, policy_text: str) -> (PolicyAnalysis, str):
        if not policy_text.strip():
            return PolicyAnalysis(), ""
        cleaned_policy = self._preprocess_policy(policy_text)
        section1, remaining = self._extract_section(cleaned_policy, "terms to avoid")
        section2, remaining = self._extract_section(remaining, "recommended terms")
        combined_section = section1 + "\n" + section2 if section1 or section2 else ""
        if not combined_section.strip():
            return PolicyAnalysis(), cleaned_policy
        analysis = self._analyze_with_openai(combined_section)
        return analysis, remaining

def clean_markdown(text: str) -> str:
    return re.sub(r'<!--\s*🖼️❌ Image not available.*?-->', "", text, flags=re.DOTALL)

def parse_sections(lines: List[str]) -> List[str]:
    sections = []
    current_section_lines = []
    for line in lines:
        if re.match(r"\s*#+\s+", line):
            if current_section_lines:
                sections.append("\n".join(current_section_lines).strip())
            current_section_lines = [line]
        else:
            current_section_lines.append(line)
    if current_section_lines:
        sections.append("\n".join(current_section_lines).strip())
    return sections

def chunk_sections(sections: List[str], max_chunk_size: int) -> List[str]:
    chunks = []
    current_chunk = ""
    for section in sections:
        sec_len = len(section)
        cur_len = len(current_chunk)
        if cur_len == 0:
            current_chunk = section
        elif cur_len + 1 + sec_len > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n" + section
        if len(current_chunk) > max_chunk_size and cur_len == 0:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def chunk_markdown_file(md_path: Path, max_chunk_size: int = 500) -> List[str]:
    with open(md_path, encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = clean_markdown(raw_text)
    lines = cleaned_text.splitlines()
    first_heading_index = None
    for idx, line in enumerate(lines):
        if re.match(r"\s*#+\s+", line):
            first_heading_index = idx
            break
    chunks = []
    if first_heading_index is not None and first_heading_index > 0:
        pre_chunk = "\n".join(lines[:first_heading_index]).strip()
        if pre_chunk:
            chunks.append(pre_chunk)
        remaining_lines = lines[first_heading_index:]
    else:
        return [cleaned_text.strip()]
    sections = parse_sections(remaining_lines)
    section_chunks = chunk_sections(sections, max_chunk_size)
    chunks.extend(section_chunks)
    return chunks

def find_violations(user_content: str, terms_to_avoid: List[str]) -> str:
    violations = []
    for term in terms_to_avoid:
        try:
            term = term.replace("\\\\", "\\")
            if "[your brand]" in term:
                pattern = re.escape(term)
                pattern = pattern.replace("\\[your\\ brand\\]", r"(?:\w+(?:\s+\w+)*)")
            else:
                base_term = re.escape(term)
                if base_term.endswith("s"):
                    pattern = base_term[:-1] + "s?"
                else:
                    pattern = base_term + "s?"
            pattern = fr"\b{pattern}\b"
            regex = re.compile(pattern, flags=re.IGNORECASE)
            for match in regex.finditer(user_content):
                violation_term = match.group()
                start_pos = max(0, match.start() - 20)
                end_pos = min(len(user_content), match.end() + 20)
                context = user_content[start_pos:end_pos].strip()
                violations.append({
                    "policy_phrase": context,
                    "violation_phrase": violation_term,
                    "explanation": "Exact term match"
                })
        except re.error as error:
            print(f"Error compiling regex for term '{term}': {str(error)}")
            continue
    return json.dumps(violations, indent=2)

async def process_message(client: AsyncOpenAI, message: str, policy_text: str) -> str:
    SYSTEM_PROMPT = f"""You are tasked with identifying policy violations in the given content.
                    Analyze the content against each policy rule and identify violations.
                    Return only a JSON array with each violation containing the exact policy phrase and the matching content phrase.
                    The phrases must be exact substrings that can be found in both the policy and content. 
                    Do give a short supporting explanation inside 'explanation'.
                    Do not assume or draw any additional implications – just analyze based on the given info.
                    Do NOT include any additional text, markdown formatting, or commentary.
                    Return ONLY the JSON array and nothing else.

                    Example:
                    [{{
                        "policy_phrase": "",
                        "violation_phrase": "",
                        "explanation": ""
                    }}]

                    POLICY:
                    ```
                    {policy_text}
                    ```
                    """
    try:
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"USER CONTENT:\n```\n{message}\n```"},
        ]
        response = await client.chat.completions.create(
            model=Config.COMPLEX_MODEL,
            messages=msg,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as error:
        print(f"Error in process_message: {str(error)}")
        return ""

def parse_json(text):
    try:
        match = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if not match:
            return None
        json_string = match.group(0)
        cleaned_json = " ".join(json_string.split())
        return json.loads(cleaned_json)
    except Exception:
        return None

@app.post("/check-compliance", response_model=List[Violation])
async def check_compliance(request: PolicyRequest):
    """Main endpoint for compliance checking."""
    policy_path = None
    content_path = None
    # Run docling for both URLs using our new pythonic approach.
    policy_path = await run_docling(request.policy_url, "policy")
    content_path = await run_docling(request.content_url, "content")

    # Load markdown files.
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_content = f.read()
    with open(content_path, "r", encoding="utf-8") as f:
        content_content = f.read()

    # Run policy analysis.
    analyzer = PolicyAnalyzer(sync_client)
    result, rem = analyzer.analyze(policy_content)

    # Level 1: Simple regex-based violations check.
    terms_to_avoid = result.terms_to_avoid
    level_one = json.loads(find_violations(content_content, terms_to_avoid))

    # Prepare for level 2 analysis.
    policy_text = "\n\n".join(result.complex_rules) + "\n\n" + rem
    chunks = chunk_markdown_file(content_path, Config.MAX_CHUNK_SIZE)

    level_two = []
    async def process_chunk(chunk: str):
        response = await process_message(async_client, chunk, policy_text)
        json_data = parse_json(response)
        if json_data and json_data != [] and json_data[0]["policy_phrase"] != "" and json_data[0]["violation_phrase"] != "":
            level_two.extend(json_data)
    await asyncio.gather(*(process_chunk(chunk) for chunk in chunks))

    total_violations = [Violation(**v) for v in level_one + level_two]    
    return total_violations
