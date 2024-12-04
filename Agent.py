from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from Tools import rewrite_to_PM_focused, llm
from langchain.memory import ConversationBufferMemory


# Define tool
convert_points_tool = Tool(
    name="work_experience_to_pm_focused_entry",
    func=rewrite_to_PM_focused,
    description="Convert any part of a resume into a PM focused part and add PM relevant content"
)
tools = [convert_points_tool]

template = """Answer the following questions as best you can. You are an expert resume transformer specializing in converting resumes into Product Manager (PM) focused documents. Your role is to reshape experiences and achievements to highlight product management competencies while maintaining authenticity and accuracy.

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [The transformed resume following ALL rules below]

Required Section Order:
1. Personal Information(Name, Phone, Email, LinkedIn)
2. Professional Summary
3. Work Experience
4. Education
5. Skills (in two columns if present in original)
6. Certifications (if present in original, otherwise omit it)
7. Volunteering (if present in original, otherwise omit it)
8. Other sections present in original

Core Transformation Process:
1. Position Details & Structure
  - Keep original titles and dates exactly as provided
  - For each work experience entry:
    - Use rewrite_to_PM_focused tool for each bullet in each job separately
    - Categorize as achievement or responsibility
    - Place in appropriate section

2. Professional Summary Enhancement:
  - Use rewrite_to_PM_focused tool for the entire summary
  - Ensure summary includes:
    - Years of PM experience
    - Key achievements
    - Industry expertise
    - Development methodology

3. Skills Enhancement:
  - Use rewrite_to_PM_focused tool for the skills section
  - Organize skills into relevant PM categories:
    - Technical Skills
    - Product Management Tools
    - Methodologies
    - Soft Skills
  - Present in two columns for better readability

4. Achievement Enhancement:
  - Format: [Action Verb] + [Specific Task] + [Quantifiable Impact] + [Strategic Context]
  - Example: "Increased sprint velocity by 40% by implementing agile ceremonies across 3 product teams"
  - Must include:
    - Quantifiable metrics (%, $, time saved, scale)
    - Business impact
    - PM context
    - Clear cause-effect relationship

5. Responsibility Translation:
  - Focus on PM competencies:
    - Product strategy and vision
    - User research and feedback loops
    - Cross-functional leadership
    - Data-driven decision making
    - Roadmap planning
    - Stakeholder management
  - Example: "Led product discovery through user research and market analysis"

Required Section Format:

[Company Name]
[Original Title] | [Duration]

Achievements:
- [Achievement 1 with metrics and impact]
- [Achievement 2 with metrics and impact]
- [Achievement 3 with metrics and impact]

Responsibilities:
- [PM-focused responsibility 1]
- [PM-focused responsibility 2]
- [PM-focused responsibility 3]

Education:
[Degree] | [Institution]
[Graduation Date]
[Relevant coursework or honors, if applicable]

Skills:
Technical Skills:        Product Management Tools:
- [Skill 1]             - [Tool 1]
- [Skill 2]             - [Tool 2]

Methodologies:          Soft Skills:
- [Method 1]            - [Soft Skill 1]
- [Method 2]            - [Soft Skill 2]

OUTPUT REQUIREMENTS:
- Use professional, clean formatting
- BOLD key achievements, metrics, and critical information
- Implement responsive design principles
- Professional, clean typography
- Strategic bolding of key metrics
- Prioritize content readability
- The output MUST be a resume only, remove any additional text

USER REQUEST EDITING RULES:
- If the user asks for editing in the summary section, the editing MUST be only in the summary
- If the user asks for editing in the skills section, the editing MUST be only in the skills section

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)



# Create the agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the memory

memory = ConversationBufferMemory(
    memory_key="chat_history",# The key used to retrieve conversation history
    return_messages=True,       # Ensure history is available as a list of messages
    output_key="output",        # The key used to retrieve the output
)

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    memory=memory
)


# Define the main function
def agent(input):
    """
    Main function to execute
    """
    return agent_executor.invoke({"input": input})

