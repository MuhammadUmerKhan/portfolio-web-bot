from langchain_core.messages import SystemMessage, AIMessage
from app.agents.state import AgentState

def get_responder_node(chatbot_instance):
    """
    Returns the final answer generator node function.
    Reads retrieved documents and knowledge graph context from state
    to produce Umer's response.
    """
    from langsmith import traceable
    
    @traceable(name="Agent Responder Node")
    def generate_response(state: AgentState) -> dict:
        # Since planner handles conversational direct replies, this node is only
        # reached if retrieval happened. No need to check tool_calls.

        # 1. Compile retrieval contexts
        context_parts = []
        if state.get("graph_context"):
            context_parts.append(f"Lightweight Knowledge Graph Context:\n{state['graph_context']}")
        if state.get("retrieved_docs"):
            context_parts.append("Retrieved Document Context:")
            for idx, doc in enumerate(state["retrieved_docs"]):
                context_parts.append(f"Doc {idx+1}:\n{doc['page_content']}")
                
        context = "\n\n".join(context_parts)

        # 2. Extract user's last query
        question = ""
        for msg in reversed(state["messages"]):
            if msg.type == "human":
                question = msg.content
                break

        # 3. Format Umer's Persona and prompt rules
        prompt_template = """
        You are Muhammad Umer Khan — a professional, polite, and passionate AI Engineer 🤖 dedicated to clear, accurate, and helpful communication.
            ✅ Only answer what is **explicitly asked** in the question — avoid extra or unrelated details.
            📝 Paraphrase from the context in your own words, keeping answers short and easy to read (1–3 sentences).
            🔹 Use bullet points only when listing multiple items, skills, experiences, or contact details, make use of emoji in response.
            🚫 If the answer is **not found** in the provided context, reply politely with:
            "I'm sorry, that information isn't available in my current context. 😊 Feel free to ask about my skills, projects, or how to contact me."
            💬 Always keep the tone clear, friendly, and professional, with light use of emojis for a human touch.
                        
            📬 If the question is about contacting you, respond with:
                "You can reach me at:
                - Phone: +923432187868 📞
                - Email: muhammadumerk546@gmail.com 📧
                - LinkedIn: https://www.linkedin.com/in/muhammad-umer-khan-61729b260/ 🔗"

            Context: {context}

            Question: {question}
            Answer:
        """

        system_prompt = SystemMessage(content=prompt_template.format(
            context=context or "No matching context found.",
            question=question
        ))

        # Invoke the ChatGroq model with system instructions + full history
        messages = [system_prompt] + list(state["messages"])
        response = chatbot_instance.llm.invoke(messages)
        
        return {"messages": [response]}
        
    return generate_response
