from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import List, Dict, Union # Added Union for chat parts

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# FastAPI app initialization
app = FastAPI()

# CORS setup (important for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Keep this for now; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response models (existing)
class PromptRequest(BaseModel):
    task_description: str
    category: str

class RefinementResponse(BaseModel):
    optimized_prompt: str
    explanation: List[str] = None
    suggestions: List[str] = None
    original_task_for_mod: str = None
    original_category_label_for_mod: str = None
    current_category_id: str = None

class ModifyPromptRequest(BaseModel):
    current_refined_prompt: str
    user_modification_instructions: str
    original_task_for_context: str
    original_category_label_for_context: str
    current_category_id: str

# NEW: Models for Chatbot interaction
class ChatMessagePart(BaseModel):
    text: str # Gemini usually uses 'text' for parts in JSON from/to frontend

class ChatMessageContent(BaseModel):
    role: str # 'user' or 'model'
    parts: List[ChatMessagePart]

class ChatRequest(BaseModel):
    history: List[ChatMessageContent] # The full conversation history

class ChatResponse(BaseModel):
    model_response: str # The new response from the LLM

# Prompt categories (KEEP THIS EXACTLY AS IS)
prompt_categories = [
    {"id": "writing", "label": "Writing & Content Creation", "template": "You are an expert prompt engineer specializing in writing. Convert the user's informal or vague input into a detailed, clear, and structured writing prompt that includes style, tone, target audience, format, and length as needed to generate high-quality content. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "ideation", "label": "Ideation & Brainstorming", "template": "You are a top prompt engineer for idea generation. Transform the user's informal or unclear input into a powerful and creative ideation prompt that specifies the domain, objective, and desired number of ideas to maximize novelty and practicality. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "summarization", "label": "Summarization & Extraction", "template": "You are an expert prompt engineer for summarization tasks. Take the user's vague input and rewrite it into a clear and specific summarization prompt that instructs extracting key points or bullet summaries in an appropriate tone and format. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "explanation", "label": "Explanation & Tutoring", "template": "You are a skilled educational prompt engineer. Convert the user's brief or unclear input into a well-structured prompt that requests a step-by-step, clear and concise explanation of the topic for someone with no prior knowledge, ensuring clarity and comprehension. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "reasoning", "label": "Problem Solving & Reasoning", "template": "You are a prompt engineer expert in logical reasoning and problem solving. Turn the user's informal question into a precise prompt that guides step-by-step reasoning, calculations, and logical conclusions. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "coding", "label": "Code & Development", "template": "You are the world's leading prompt engineer for software development. Transform the user's informal coding request into a specific, detailed prompt that defines the programming language, required functionality, constraints, expected outputs, and error handling. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "data_analysis", "label": "Data & Analysis", "template": "You are an expert prompt engineer for data analysis. Rewrite the user's unclear or partial request into a precise prompt that clearly states the analytical goal, data type, and desired output format such as summaries, formulas, or queries. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "language", "label": "Translation & Language", "template": "You are a language prompt engineer. Convert the user's casual translation or style request into an exact prompt specifying language pairs, tone or style changes, formality level, and any special vocabulary requirements. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "roleplay", "label": "Roleplay & Simulation", "template": "You are a specialist prompt engineer for immersive roleplay and simulation. Turn the user's vague input into a detailed prompt defining the scenario, roles, objectives, tone, and boundaries to create an engaging simulation experience. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "research", "label": "Research & Reports", "template": "You are an expert prompt engineer for research and professional reports. Rewrite the user's informal query into a structured prompt specifying topic scope, report type, depth, formatting style, and references if applicable. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "productivity", "label": "Productivity & Planning", "template": "You are a prompt engineer specializing in productivity and planning. Convert the user's casual goals into an actionable prompt that generates clear plans, timelines, schedules, or task breakdowns suited to their context. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "visual_generation", "label": "Image/Media Generation", "template": "You are the world's best prompt engineer for visual content generation. Transform the user's rough image idea into a rich, detailed prompt that includes subject details, style, lighting, mood, background, and composition instructions. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "expert_domains", "label": "Legal, Medical, and Expert Domains", "template": "You are a specialist prompt engineer for expert domains such as law and medicine. Turn the user's vague or casual input into a precise, context-aware prompt suitable for AI-assisted expert content generation. Provide only the final refined prompt as the output, without any explanations or extra text. Input: {{user_input}}"},
    {"id": "meta_prompting", "label": "Meta-Prompts (Prompt Refinement & Generation)", "template": "You are the world's best prompt engineer assistant. Transform any unclear or informal prompt into a detailed, optimized version that adds context, format, tone, constraints, and purpose to maximize AI quality.\n\nUser Input: {{user_input}}\n\nRefined Prompt:"}
]

NO_SELECTION_CATEGORY_ID = "please_select"
default_prompt_category = {
    "id": "meta_prompting",
    "label": "Meta-Prompts (Prompt Refinement & Generation)",
    "template": "You are the world's best prompt engineer assistant. Transform any unclear or informal prompt into a detailed, optimized version that adds context, format, tone, constraints, and purpose to maximize AI quality.\n\nUser Input: {{user_input}}\n\nRefined Prompt:"
}

def get_prompt_category(category_id: str):
    for category in prompt_categories:
        if category["id"] == category_id:
            return category
    return default_prompt_category

def get_category_prompt(category_id: str, user_input: str) -> str:
    category_info = get_prompt_category(category_id)
    return category_info["template"].replace("{{user_input}}", user_input)

def parse_bullet_list(text_content: str) -> List[str]:
    if not text_content:
        return []
    lines = text_content.strip().split('\n')
    parsed_items = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('* '):
            parsed_items.append(stripped_line[2:].strip())
        elif stripped_line:
            parsed_items.append(stripped_line)
    return parsed_items if parsed_items else [text_content.strip()]

def modify_prompt_with_user_input(
    current_refined_prompt: str,
    user_modification_instructions: str,
    original_task_for_context: str,
    original_category_label_for_context: str
) -> str:
    modification_prompt = f"""
    You are an expert prompt engineer.
    The user has already provided an initial task: "{original_task_for_context}" and chosen the category: "{original_category_label_for_context}".
    Based on this, a refined prompt was previously generated.

    Now, the user wants to make specific further modifications to that refined prompt.
    Take the 'CURRENT REFINED PROMPT' and apply the 'USER MODIFICATION INSTRUCTIONS'.
    Ensure the output is only the single, newly modified prompt, without any conversational filler or introductory/concluding text.

    CURRENT REFINED PROMPT:
    {current_refined_prompt}

    USER MODIFICATION INSTRUCTIONS:
    {user_modification_instructions}
    """
    try:
        response = model.generate_content(modification_prompt)
        return response.text.strip() if response and hasattr(response, "text") else current_refined_prompt
    except Exception as e:
        print(f"Error modifying prompt with user input: {e}")
        return current_refined_prompt

def get_ai_explanation_and_suggestions(generated_prompt: str, original_task: str, category_label: str) -> Dict[str, List[str]]:
    explanation_prompt = f"""
    You are an expert prompt engineer.
    Explain why the following prompt was generated in this way, considering the original user task: "{original_task}" and the category: "{category_label}".
    Highlight the key elements added or modified to make the prompt more effective for an AI.
    Provide only the explanation as a concise bulleted list with a maximum of 3-4 key points, starting each point with an asterisk (*). Do not include any introductory or concluding sentences.

    Generated Prompt:
    {generated_prompt}
    """

    suggestion_prompt = f"""
    You are an expert prompt engineer.
    Based on the following generated prompt and the original user task: "{original_task}", suggest 2-3 actionable ways the user could have phrased their initial request to potentially get an even better or more directly relevant prompt in the future.
    Provide only the suggestions as a concise bulleted list, starting each point with an asterisk (*). Do not include any introductory or concluding sentences.

    Generated Prompt:
    {generated_prompt}
    """

    explanation_response = model.generate_content(explanation_prompt)
    suggestion_response = model.generate_content(suggestion_prompt)

    explanation_raw = explanation_response.text.strip() if explanation_response and hasattr(explanation_response, 'text') else ""
    suggestions_raw = suggestion_response.text.strip() if suggestion_response and hasattr(suggestion_response, 'text') else ""

    explanation = parse_bullet_list(explanation_raw)
    suggestions = parse_bullet_list(suggestions_raw)

    return {"explanation": explanation, "suggestions": suggestions}


@app.get("/api/categories", response_model=List[Dict[str, str]])
async def get_categories():
    return [{"id": c["id"], "label": c["label"]} for c in prompt_categories]


@app.post("/api/generate-prompt", response_model=RefinementResponse)
async def generate_prompt_api(data: PromptRequest):
    try:
        selected_category_id = data.category if data.category != NO_SELECTION_CATEGORY_ID else default_prompt_category["id"]
        category_info = get_prompt_category(selected_category_id)

        prompt_text = get_category_prompt(selected_category_id, data.task_description)
        response = model.generate_content(prompt_text)
        refined_prompt = response.text.strip() if response and hasattr(response, "text") else ""

        ai_analysis = get_ai_explanation_and_suggestions(
            refined_prompt, data.task_description, category_info["label"]
        )

        return RefinementResponse(
            optimized_prompt=refined_prompt,
            explanation=ai_analysis["explanation"],
            suggestions=ai_analysis["suggestions"],
            original_task_for_mod=data.task_description,
            original_category_label_for_mod=category_info["label"],
            current_category_id=selected_category_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/modify-prompt", response_model=RefinementResponse)
async def modify_prompt_api(data: ModifyPromptRequest):
    try:
        modified_prompt = modify_prompt_with_user_input(
            data.current_refined_prompt,
            data.user_modification_instructions,
            data.original_task_for_context,
            data.original_category_label_for_context
        )

        ai_analysis = get_ai_explanation_and_suggestions(
            modified_prompt,
            data.original_task_for_context,
            data.original_category_label_for_context
        )

        return RefinementResponse(
            optimized_prompt=modified_prompt,
            explanation=ai_analysis["explanation"],
            suggestions=ai_analysis["suggestions"],
            original_task_for_mod=data.original_task_for_context,
            original_category_label_for_mod=data.original_category_label_for_context,
            current_category_id=data.current_category_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW API endpoint for interactive chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(data: ChatRequest):
    try:
        if not data.history:
            raise HTTPException(status_code=400, detail="Chat history cannot be empty.")

        # Transform frontend history format to Gemini's expected format
        gemini_history = []
        for msg in data.history:
            # Assuming each part is a single text string for simplicity
            gemini_history.append({
                "role": msg.role,
                "parts": [{"text": p.text} for p in msg.parts]
            })

        # Send the full history to Gemini
        # Using generate_content with contents=history handles the context
        response = model.generate_content(contents=gemini_history)

        model_response = response.text.strip() if response and hasattr(response, "text") else "No response from AI."
        return ChatResponse(model_response=model_response)
    except Exception as e:
        print(f"Error in chat API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat response: {str(e)}")