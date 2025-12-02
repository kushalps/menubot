import os
import json
import random
from typing import TypedDict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# ✅ New Ollama integration (no OpenAI, no bind_tools)
from langchain_ollama import ChatOllama

#Load environment

load_dotenv()


#Family profile tuning

FAMILY_PROFILE = {
    "avoid_ingredients": ["baingan", "karela"],  # things family doesn’t like
    "prefer_low_oil": True,
    "prefer_millets": True,
}

#LLM base (local phi3 via Ollama)

llm = ChatOllama(
    model="phi3", 
    temperature=0.4,
)

#LangGraph State

class ChatState(TypedDict):
    messages: List[dict]  # [{"role": "user"/"assistant", "content": "..."}]


# Tool: North Indian Menu Planner

class MenuRequest(BaseModel):
    meal_type: Literal["breakfast", "lunch", "dinner", "snacks", "full_day"] = Field(
        default="full_day",
        description="Which meal to suggest."
    )
    veg_or_nonveg: Literal["veg", "nonveg", "mixed"] = Field(
        default="veg",
        description="Diet preference."
    )
    spice_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Preferred spice level."
    )
    health_focus: Literal["normal", "light", "diabetic_friendly", "low_oil"] = Field(
        default="normal",
        description="Health preference."
    )
    extra_notes: Optional[str] = Field(
        default=None,
        description="Dislikes, time constraints, etc."
    )


@tool("north_indian_menu_planner", args_schema=MenuRequest)
def north_indian_menu_planner(
    meal_type: str,
    veg_or_nonveg: str,
    spice_level: str,
    health_focus: str,
    extra_notes: Optional[str] = None,
):
    """
    Suggests a North-Indian, home-style menu ,
    tuned to a typical family: low oil, simple, realistic dishes.
    """

    avoid_words = [w.lower() for w in FAMILY_PROFILE.get("avoid_ingredients", [])]

    def keep_dish(name: str) -> bool:
        lower = name.lower()
        return not any(word in lower for word in avoid_words)

    def dish_line(name, notes=""):
        if notes:
            return f"- {name} – {notes}"
        return f"- {name}"

    breakfast_options_veg = [
        dish_line("Vegetable poha", "light, add peanuts, coriander, lemon"),
        dish_line("Aloo paratha with curd", "comfort food, use less oil/ghee"),
        dish_line("Besan cheela with green chutney", "good protein, low oil"),
        dish_line("Upma with vegetables", "light, quick, less oil"),
        dish_line("Moong dal cheela", "very good protein and light"),
    ]

    lunch_options_veg = [
        dish_line("Dal tadka (arhar or moong) + jeera rice + salad"),
        dish_line("Rajma chawal + cucumber-onion salad"),
        dish_line("Mixed veg (gajar-matar, beans, gobhi) + phulka + raita"),
        dish_line("Chole + jeera rice", "slightly heavier, make with less oil"),
        dish_line("Lauki chana dal + roti + simple kachumber salad"),
    ]

    dinner_options_veg = [
        dish_line("Palak paneer + phulka + salad"),
        dish_line("Masoor dal + jeera rice + bhindi sabzi (low oil)"),
        dish_line("Khichdi (moong dal) + curd + papad", "very light, good for night"),
        dish_line("Tori/tinda sabzi + phulka + raita"),
    ]

    snack_options = [
        dish_line("Roasted makhana with ghee and black salt"),
        dish_line("Sprouts chaat with onion, tomato, coriander, lemon"),
        dish_line("Fruit chaat (apple, banana, pomegranate, black salt)"),
        dish_line("Masala chai + 2 marie biscuits"),
        dish_line("Home-made bhel with murmura, onion, tomato, coriander"),
    ]

    if FAMILY_PROFILE.get("prefer_millets", False):
        breakfast_options_veg.append(
            dish_line("Ragi / jowar cheela", "great for breakfast, use curd + chutney")
        )
        dinner_options_veg.append(
            dish_line("Jowar / bajra roti + sabzi + curd", "good especially in winters")
        )

    nonveg_add = [
        dish_line("Egg bhurji with onion-tomato, less oil"),
        dish_line("Simple chicken curry with onion-tomato base (home style)"),
        dish_line("Light fish curry with tomato-mustard base"),
    ]

    health_note = ""
    if health_focus == "light" or FAMILY_PROFILE.get("prefer_low_oil", False):
        health_note += (
            "Use minimal oil (brush or 1–2 tsp), avoid deep-frying, "
            "and prefer boiling/steaming/sauteing instead of frying.\n"
        )
    if health_focus == "diabetic_friendly":
        health_note += (
            "Keep rice portion small, prefer phulka without ghee, avoid potato/sugar, "
            "and add extra salad and leafy vegetables.\n"
        )
    if health_focus == "low_oil":
        health_note += "Avoid pakoras, bhaturas and deep-fried snacks as far as possible.\n"

    random.seed(42)

    def choose_two(items):
        filtered = [x for x in items if keep_dish(x)]
        if len(filtered) <= 2:
            return filtered
        return random.sample(filtered, 2)

    menu_lines = [
        "Here is a simple North-Indian home-style menu for your family:",
        f"Diet: {veg_or_nonveg}, Spice level: {spice_level}"
    ]

    if extra_notes:
        menu_lines.append(f"Extra notes considered: {extra_notes}")

    if health_note:
        menu_lines.append("Health notes: " + health_note)

    def add_meal_block(title, items):
        menu_lines.append(f"\n{title}:")
        for d in items:
            menu_lines.append(d)
        menu_lines.append(
            "Cooking style tip: use onion-tomato-ginger(-garlic) base, "
            "roast masalas on low flame, keep oil minimal, adjust salt and chilli as per taste."
        )

    if meal_type in ["breakfast", "full_day"]:
        add_meal_block("Breakfast", choose_two(breakfast_options_veg))
    if meal_type in ["lunch", "full_day"]:
        add_meal_block("Lunch", choose_two(lunch_options_veg))
    if meal_type in ["dinner", "full_day"]:
        add_meal_block("Dinner", choose_two(dinner_options_veg))
    if meal_type in ["snacks", "full_day"]:
        add_meal_block("Snacks / Evening", choose_two(snack_options))

    if veg_or_nonveg in ["nonveg", "mixed"]:
        menu_lines.append("\nNon-veg option(s) you can add:")
        for d in choose_two(nonveg_add):
            menu_lines.append(d)

    menu_lines.append(
        "\nTip: Check what sabzi is actually available at home and how tired you are, "
        "then pick the lighter/easier option from above."
    )

    return "\n".join(menu_lines)


#Heuristic: decide when & how to use the tool

def infer_menu_request(user_text: str) -> Optional[MenuRequest]:
    """
    Very simple heuristic to convert free-text into MenuRequest.
    If it can't detect anything meaningful, returns None.
    """
    t = user_text.lower()

    # Only trigger for menu-ish queries
    if not any(k in t for k in ["menu", "breakfast", "lunch", "dinner", "snack", "food", "khana", "khaana"]):
        return None

    # meal_type
    if "breakfast" in t or "nashta" in t or "nashta" in t:
        meal_type = "breakfast"
    elif "lunch" in t:
        meal_type = "lunch"
    elif "dinner" in t or "raat" in t or "night" in t:
        meal_type = "dinner"
    elif "snack" in t or "evening" in t or "chai" in t:
        meal_type = "snacks"
    elif "full day" in t or "whole day" in t or "entire day" in t or "today" in t:
        meal_type = "full_day"
    else:
        # default to breakfast for short simple requests like "suggest me healthy breakfast"
        if "healthy breakfast" in t:
            meal_type = "breakfast"
        else:
            meal_type = "full_day"

    # veg / nonveg
    if "nonveg" in t or "non-veg" in t or "chicken" in t or "mutton" in t or "fish" in t:
        veg_or_nonveg = "nonveg"
    elif "mixed" in t:
        veg_or_nonveg = "mixed"
    else:
        veg_or_nonveg = "veg"

    # spice level
    if "spicy" in t or "mirchi" in t or "teekha" in t:
        spice_level = "high"
    elif "less spicy" in t or "no spice" in t or "bland" in t or "not spicy" in t:
        spice_level = "low"
    else:
        spice_level = "medium"

    # health focus
    if "diabetic" in t or "sugar" in t:
        health_focus = "diabetic_friendly"
    elif "light" in t or "diet" in t or "healthy" in t:
        health_focus = "light"
    elif "low oil" in t or "less oil" in t:
        health_focus = "low_oil"
    else:
        health_focus = "normal"

    return MenuRequest(
        meal_type=meal_type,
        veg_or_nonveg=veg_or_nonveg,
        spice_level=spice_level,
        health_focus=health_focus,
        extra_notes=user_text,
    )


#Assistant node for LangGraph

def assistant_node(state: ChatState) -> ChatState:
    """
    - Look at last user message.
    - If it looks like a menu request, calls north_indian_menu_planner directly.
    - Otherwise, uses phi3 LLM for general chat.
    """

    if not state["messages"]:
        return state

    last_user_msg = None
    for m in reversed(state["messages"]):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    if last_user_msg is None:
        return state

    #Try to interpret as menu request
    menu_req = infer_menu_request(last_user_msg)

    if menu_req is not None:
        tool_result = north_indian_menu_planner.invoke(menu_req.dict())
        state["messages"].append({"role": "assistant", "content": tool_result})
        return state

    #Otherwise, use LLM for general chat
    messages = [
        SystemMessage(
            content=(
                "You are a friendly North Indian home-food assistant for an Indian college student. "
                "You help families plan realistic daily menus with simple, low-oil recipes. "
                "If user is not asking for menu, answer briefly and helpfully."
            )
        )
    ]

    for m in state["messages"]:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    ai_msg = llm.invoke(messages)
    if ai_msg.content:
        state["messages"].append({"role": "assistant", "content": ai_msg.content})

    return state


#Build LangGraph graph

graph_builder = StateGraph(ChatState)
graph_builder.add_node("assistant", assistant_node)
graph_builder.set_entry_point("assistant")
graph_builder.add_edge("assistant", END)

app = graph_builder.compile()


#Conversation save / load

def save_conversation(state: ChatState, filename: str = "conversation_history.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state["messages"], f, ensure_ascii=False, indent=2)


def load_conversation(filename: str = "conversation_history.json") -> ChatState:
    if not os.path.exists(filename):
        return {"messages": []}
    with open(filename, "r", encoding="utf-8") as f:
        msgs = json.load(f)
    return {"messages": msgs}


#Simple CLI chat loop

def run_cli_chat():
    print("Food Menu Chatbot (North Indian Home Style, local phi3)")
    print("Type 'exit' to quit.\n")

    state = load_conversation()

    if state["messages"]:
        print("(Loaded previous conversation from conversation_history.json)")
        print("-" * 40)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Bye! Happy cooking at home :)")
            save_conversation(state)
            print("Conversation saved to conversation_history.json")
            break

        state["messages"].append({"role": "user", "content": user_input})

        state = app.invoke(state)

        last_msg = state["messages"][-1]["content"]
        print("Bot:", last_msg)
        print("-" * 40)

        save_conversation(state)


if __name__ == "__main__":
    run_cli_chat()

