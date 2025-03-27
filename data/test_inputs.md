python main.py --debug DEBUG

# Predicted Outputs
This notebook will help you learn how to use Predicted Outputs.

Tutorial

Machine Learning Engineers

from openai import OpenAI

code = """
class User {
  firstName: string = "";
  lastName: string = "";
  username: string = "";
}

export default User;
"""

refactor_prompt = """
Replace the "username" property with an "email" property. Respond only 
with code, and with no markdown formatting.
"""

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": refactor_prompt
        },
        {
            "role": "user",
            "content": code
        }
    ],
    prediction={
        "type": "content",
        "content": code
    }
)

print(completion)
print(completion.choices[0].message.content)


Predicted Outputs
Reduce latency for model responses where much of the response is known ahead of time.
Predicted Outputs enable you to speed up API responses from Chat Completions when many of the output tokens are known ahead of time. This is most common when you are regenerating a text or code file with minor modifications. You can provide your prediction using the prediction request parameter in Chat Completions.

Predicted Outputs are available today using the latest gpt-4o and gpt-4o-mini models. Read on to learn how to use Predicted Outputs to reduce latency in your applicatons.



---

# Agents SDK
The notebook should help customer understand how to use Agents SDK from OpenAI to build a multiagent framework that enables to generate synthetic transcripts between a telecommunication Customer Rep and their respective customer with multiple turns imitating a real customer conversation.

The notebook should help customer understand how to use Agents SDK from OpenAI to build a multiagent framework that enables to handoff to a Math agent that has a calculator tool and a Web Searcher agent that has a tool to search the web.

Tutorial

Machine Learning Engineers

Use new agents sdk from openai