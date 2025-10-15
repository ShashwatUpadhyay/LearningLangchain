from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template="""
    Please Summarize the research paper titled {paper_input} with the following specifications:
    Explaination style: {style_input}
    Explanation Length: {length_input}
    1. Mathmatical Detail:
        - Include relevant mathmatical equations if present in the paper.
        - Explain the mathmatical concepts using simple, intutive code snippet where applicable.
    2. Analogies:
        - Use relatable analogies to simplify complex ideas
    If certain information is not available in the paper, respond with "Information not available in the paper." instead of guessing.
    Ensure the summary is clean, accurate, and aligned with the provided stule the length.
    """,
    input_variables=["paper_input", "style_input", "length_input"]
)  

template.save('template.json')