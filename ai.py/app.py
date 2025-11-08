from langchain_core.prompts import PromptTemplate

plan_prompt = PromptTemplate(
    template="""You are a planner. Your job is to decide if a user's question requires searching a knowledge base.

    If the question is asking for specific information or facts, respond with the single word 'retrieve'.

    If the question is a greeting, a general conversation starter, or something that doesn't need a knowledge base, respond with the single word 'generate'.

    Question: {question}

    Decision:""",
    input_variables=["question"],
)

