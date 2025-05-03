def design_prompt(list_of_agents, user_query):
   prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n """
   prompt += f"""You are an expert model routing agent responsible for selecting the most suitable AI agent from a given library based on query requirements. You will be provided with metadata for each agent. Your task is to analyze the input query and choose the most appropriate agent to handle it effectively.\n"""

   prompt += f"""For each AI agent, you are provided:\n
   - **Agent Name**: The model name.\n
   - **Agent Specifications**: Various specifications of the agent.\n
   - **Strengths**: Capabilities where this agent excels, with a brief description for each.\n
   - **Weaknesses**: Limitations of this agent, with a brief description for each.\n
   - **Resource and Latency**: VRAM usage and inference time for response generation.\n
   - **Optimal Use Cases**: Scenarios where this agent performs best.\n\n"""


   prompt += f"### Available Agents:\n "
   for i, agent in enumerate(list_of_agents, 1):
      prompt += f"""Agent {i}:\n
      - **Agent Name**: {agent['name']}\n 
      - **Agent Specifications**: {agent['specs']}\n
      - **Strengths**: {agent['strengths']}\n
      - **Weaknesses**: {agent['weaknesses']}\n
      - **Resource and Latency**: {agent['res_lat']}\n
      - **Optimal Use Cases**: {agent['opt_use_cases']}\n\n"""

   prompt += f"""### Decision Rules:\n
   1. Ensure optimal and logical decision-making.\n
   2. Evaluate the capabilities of all available agents and apply reasoning to select the most suitable one.\n
   3. Prioritize agents whose optimal Use cases and Strengths align best with the query requirements.\n
   4. If multiple agents are applicable, prioritize based on the **performance-resource balance** (consider VRAM and latency).\n
   5. **Output only the name of the selected agent**, with no additional explanation or characters.\n\n"""

   
   prompt += """### Output Format:
   You reply in JSON format with the field 'chosen_model'.
   Example answer: {'chosen_model': <agent_name>}
   Replace <agent_name> with the best suitable agent's name, without any additional text. <|eot_id|>\n """
   
   prompt += f"""<|start_header_id|>user<|end_header_id|>\n """
   prompt += f"""{user_query}<|eot_id|>\n """
   prompt += """<|start_header_id|>assistant<|end_header_id|> {'chosen_model': """

   print(f"Length of prompt: {len(prompt)}")
   return prompt