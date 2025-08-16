from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_core.messages import ToolMessage

llm = init_chat_model(model="openai:gpt-5-nano")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 100;"))

@tool
def sql_tool(sql_query: str):
    """
        Must call this function to execute the sql query and get the results.
    """
    
    return db.run(sql_query)
    


prompt = """
# SQL Expert Assistant Prompt

You are an expert SQL database analyst. Your role is to analyze user questions and database schemas to generate accurate SQL queries and provide comprehensive responses.

## Your Process:
1. **Analyze** the user question and database schema carefully
2. **Generate** an appropriate SQL query based on the schema
3. **Execute** the query using the `sql_tool` function
4. **Interpret** the results and provide a clear answer

## Input Parameters:
- **Database Type**: {db_type}
- **User Question**: {question}
- **Database Schema**: {db_schema}

## SQL Query Guidelines:
- Write clean, readable SQL with proper formatting
- Use appropriate JOIN types when connecting tables
- Apply WHERE clauses for filtering when needed
- Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
- Include ORDER BY for sorted results when relevant
- Limit results if the question implies a specific number (TOP N, LIMIT)
- Use proper table and column aliases for clarity
- Handle potential NULL values appropriately

## Tool Usage:
Call the `sql_tool` function with your generated SQL query. The tool will execute the query against the database and return the results.

## Edge Case Handling:
- If the user question is ambiguous, make reasonable assumptions and explain them
- If required tables/columns don't exist in the schema, mention this limitation
- If the query might return too many results, consider adding appropriate LIMIT clauses
- Handle date/time formatting based on the question context

## Response Guidelines:
- **Answer**: Provide a direct, clear response to the user's question using the query results
- **SQL Query**: Include the exact query used, properly formatted
- **Explanation**: When helpful, briefly explain the query approach or any assumptions made

Remember to be precise, accurate, and provide context that helps the user understand both the answer and how it was derived.
"""
print("User question: \n")
user_question = input("Enter the question!")
pp_template = PromptTemplate.from_template(prompt)

chain = pp_template | llm.bind_tools([sql_tool])
import json 
tool_map = {"sql_tool": sql_tool}

response = chain.invoke({"question": user_question, "db_schema": db.get_usable_table_names(), "db_type":db.dialect})
output = None
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call["id"]
    output = tool_map[tool_name].invoke(tool_args["sql_query"])
    
print("---"*100)
print(output)
print("\n\n")


drools_prompt = """
You are a Drools expert. Convert the SQL query results into business rules in Drools format.

## Input:
- **Original Question**: {question}
- **SQL Output**: {sql_output}

## Your Task:
Generate Drools (.drl) business rules based on the SQL data that can be used for:
- Data validation
- Business logic decisions  
- Conditional processing
- Automated actions

## Drools Format Requirements:
- Include package declaration
- Import relevant model classes
- Create meaningful rule names
- Use proper when/then syntax
- Add helpful comments
- Include System.out.println for debugging

## Example Structure:
```
package com.company.rules;

import com.company.model.*;

rule "Rule Name"
    when
        // conditions based on SQL data
    then
        // actions to take
        System.out.println("Rule executed");
end
```

Generate practical Drools rules that a business user could understand and modify.
"""


final_execution = PromptTemplate.from_template(drools_prompt)

chain = final_execution | llm 

drools_output = chain.invoke({"question": user_question, "sql_output":output})
print("---"*100)
print(drools_output.content)