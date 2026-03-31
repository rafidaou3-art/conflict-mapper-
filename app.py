from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


def extract_json_object(text: str):
    if not text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "[{":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Gemini did not return valid JSON. Raw response: {cleaned[:1200]}")


def build_static_prompt(text: str) -> str:
    return f"""Analyze this conflict-related text and extract:
1. All actors (countries, groups, organizations, militias, individuals)
2. Relationships between actors (allies, enemies, neutral, supports, opposes, attacks, negotiates with)
3. Brief description of each actor's role
4. Each actor's level of influence in the conflict (scale 1-10)

Text to analyze:
{text}

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
  "actors": [
    {{
      "id": "unique_id",
      "name": "Actor Name",
      "type": "government/militia/ngo/ethnic_group/individual",
      "description": "brief role",
      "influence": 7
    }}
  ],
  "relationships": [
    {{"source": "actor_id", "target": "actor_id", "type": "allied/hostile/neutral/economic/diplomatic", "description": "brief description", "intensity": 1}}
  ]
}}

Influence scale (1-10):
- 1-3: Minor actor, limited impact on conflict
- 4-6: Moderate influence, regional importance
- 7-8: Major actor, significant power and control
- 9-10: Dominant actor, controls conflict dynamics

Rules:
- Make sure each relationship source and target matches an actor id.
- Use intensity 1-3 for weak, 4-7 for moderate, 8-10 for strong relationships.
- Return only raw JSON.
"""


def build_timeline_prompt(text: str) -> str:
    return f"""Analyze this conflict-related text and extract temporal information:

1. Identify key time periods or years mentioned in the conflict
2. For each time period, identify:
   - Which actors were active
   - What relationships existed between actors
   - Major events or changes
   - Each actor's level of influence (scale 1-10)

Text to analyze:
{text}

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
  "timeline": [
    {{
      "period": "1975-1976",
      "year": 1975,
      "label": "Early Phase",
      "description": "Brief description of this period",
      "actors": [
        {{
          "id": "unique_id",
          "name": "Actor Name",
          "type": "government/militia/ngo/ethnic_group/individual",
          "description": "role in this period",
          "status": "active/emerged/weakened/dissolved",
          "influence": 8
        }}
      ],
      "relationships": [
        {{"source": "actor_id", "target": "actor_id", "type": "allied/hostile/neutral/economic/diplomatic", "description": "brief description", "intensity": 1}}
      ],
      "events": ["Major event 1", "Major event 2"]
    }}
  ]
}}

Influence scale (1-10):
- 1-3: Minor actor, limited impact
- 4-6: Moderate influence, regional importance
- 7-8: Major actor, significant power
- 9-10: Dominant actor, controls conflict dynamics

Rules:
- Create 3-8 time periods covering the conflict's evolution when possible.
- Make sure relationship source and target match actor IDs within each period.
- Return only raw JSON.
"""


def sanitize_static_result(result: dict) -> dict:
    actors = result.get("actors", []) if isinstance(result, dict) else []
    relationships = result.get("relationships", []) if isinstance(result, dict) else []

    valid_actor_ids = set()
    cleaned_actors = []

    for actor in actors:
        if not isinstance(actor, dict):
            continue
        actor_id = actor.get("id") or actor.get("name")
        actor_name = actor.get("name") or actor_id
        if not actor_id or not actor_name:
            continue
        cleaned_actor = {
            "id": str(actor_id),
            "name": str(actor_name),
            "type": str(actor.get("type", "individual")),
            "description": str(actor.get("description", "")),
            "influence": max(1, min(10, int(actor.get("influence", 5) or 5))),
        }
        if "status" in actor:
            cleaned_actor["status"] = str(actor.get("status", "active"))
        cleaned_actors.append(cleaned_actor)
        valid_actor_ids.add(cleaned_actor["id"])

    cleaned_relationships = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source")
        target = rel.get("target")
        if source in valid_actor_ids and target in valid_actor_ids and source != target:
            cleaned_relationships.append({
                "source": str(source),
                "target": str(target),
                "type": str(rel.get("type", "neutral")),
                "description": str(rel.get("description", "")),
                "intensity": max(1, min(10, int(rel.get("intensity", 5) or 5))),
            })

    return {"actors": cleaned_actors, "relationships": cleaned_relationships}


def sanitize_timeline_result(result: dict) -> dict:
    timeline = result.get("timeline", []) if isinstance(result, dict) else []
    cleaned_timeline = []

    for period in timeline:
        if not isinstance(period, dict):
            continue
        sanitized_period = sanitize_static_result({
            "actors": period.get("actors", []),
            "relationships": period.get("relationships", []),
        })
        cleaned_timeline.append({
            "period": str(period.get("period", "")),
            "year": period.get("year"),
            "label": str(period.get("label", period.get("period", ""))),
            "description": str(period.get("description", "")),
            "actors": sanitized_period["actors"],
            "relationships": sanitized_period["relationships"],
            "events": [str(e) for e in period.get("events", []) if str(e).strip()],
        })

    return {"timeline": cleaned_timeline}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    include_timeline = bool(data.get("include_timeline", False))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prompt = build_timeline_prompt(text) if include_timeline else build_static_prompt(text)

    try:
        response = model.generate_content(prompt)
        response_text = (getattr(response, "text", "") or "").strip()

        print("\\n=== RAW GEMINI RESPONSE START ===")
        print(response_text[:3000] if response_text else "[EMPTY RESPONSE]")
        print("=== RAW GEMINI RESPONSE END ===\\n")

        parsed = extract_json_object(response_text)
        result = sanitize_timeline_result(parsed) if include_timeline else sanitize_static_result(parsed)

        if include_timeline and not result.get("timeline"):
            return jsonify({
                "error": "Gemini response was parsed, but no timeline data was found.",
                "raw_response": response_text[:1200]
            }), 500

        if not include_timeline and not result.get("actors"):
            return jsonify({
                "error": "Gemini response was parsed, but no actor data was found.",
                "raw_response": response_text[:1200]
            }), 500

        return jsonify(result)

    except Exception as e:
        print("\\n=== ANALYZE ERROR ===")
        print(str(e))
        print("=== END ANALYZE ERROR ===\\n")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
