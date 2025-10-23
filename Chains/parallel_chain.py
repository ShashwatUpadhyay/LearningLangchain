from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or another supported model name
    temperature=0.7,
    max_tokens=None,  # use default or set a limit
    timeout=None,
    max_retries=2
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided noted and quiz into a single document \n noted -> {notes} \n quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser ,
    'quiz' : prompt2 | model | parser
})

marge_chain = prompt3 | model | parser

chain = parallel_chain | marge_chain

text = """
Relational data Model:Relational data model concepts, integrity constraints: entity
integrity, referential integrity, Keys constraints, Domain constraints.
Relational Algebra: Cartesian product, Union, Intersection, Difference, Select Operation,
Project Operation, Composition of Select and Project operations, rename, Join operation.
Relational Data Model in DBMS
DBMS provides the different types of model to the user; the relational model is the one type of model that is
provided by the DBMS. The relational model is used to represent how we can store the data in relational databases.
Basically, relational databases store the data in table relations that means column and rows format. Every row and
column of the table collects the records or data that are related to the table values. The table name and column
name is useful to determine the meaning of each row from the table.
1. Basic Concept in Relational Model
2. Attribute: Attributes means each column name from the table. By using attributes, we can define the
relation.
3. Tables: All data and records we store into the tabular format in a relational model that we call table.
Every table contains rows and columns that we also call properties of the table.
4. Tuple: Tuple means a single row from the specified table that contains the single value.
5. Relational Schema: Relational schema means the name of the relation with its attribute.
6. Degree: How many attributes present in the relation that we called the degree of the relation.
7. Column: It is a value of a specified attribute.
8. Relation Instance: A relation instance means a set of finite tuples, and it never has a duplicate value.
9. Relation Key: Each row has more than one attribute, which we called relation key.
10. Attribute Domain: Attribute domain means predefined value and scope of an attribute.
2. Relational Integrity Constraints
Relation integrity constraint is used in DBMS for condition, and that condition must be present for the valid
relation. These Relational constraints in DBMS are obtained from the standards in the small world that the
information represents. Thus, there are numerous sorts of Integrity Constraints in DBMS.
Basically, relation constraints are classified into the three types as follows:
• Domain Constraints: Domain constraint can be abused if specified attribute value not corresponding to
the domain, or we can say that not in a specified data type. Domain constraint indicates that inside each
tuple, and the estimation of each quality should be one of a kind. This is determined as information types
which incorporate standard information type’s numbers, real numbers, characters, booleans, variablelength
strings.
• Key Constraints: The key constraint is the most important constraint in the relation model. By using
key constraint, we can uniquely identify the rows in the table.
• Referential Integrity Constraints: It is based on the foreign key concept; the foreign key is an essential
attribute in relation models because it relates between two different tables. For referential integrity, we
must need key attributes in the table.
3. Operations in Relational Model
We can perform the different operations on relational database models as follows:
• Insert: By using the insert operation, we can insert records into the table or say relation.
• Delete: By using the delete operation, we can remove the records from the table.
"""
#LCEL
# chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'text':text})

chain.get_graph().print_ascii()
print(result)