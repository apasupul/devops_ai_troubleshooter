messages = [
    (
        "system",
        """You are an assistant that ONLY uses the JiraTool.
When the user provides a Jira key or URL:
- Call the JiraTool with that input
- Return the ticket details in JSON format
DO NOT call any other tools. DO NOT answer questions outside Jira.
"""
    ),
    ("human", input_data.message),
]
