import re
import uuid
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    USER_ID = "user_id"
    TICKET_ID = "ticket_id"
    CUSTOM = "custom"

@dataclass
class PIIEntity:
    original_value: str
    anonymized_value: str
    pii_type: PIIType
    context: Optional[str] = None

class PIIAnonymizer:
    """
    Comprehensive PII anonymization system that detects, anonymizes, stores mappings,
    and can deanonymize data for MCP server responses.
    """
    
    def __init__(self):
        self.entity_map: Dict[str, PIIEntity] = {}
        self.reverse_map: Dict[str, str] = {}  # anonymized -> original
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Initialize regex patterns for different PII types"""
        return {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            PIIType.SSN: re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            # Common name patterns (basic - you may want to enhance this)
            PIIType.NAME: re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            # JIRA ticket IDs
            PIIType.TICKET_ID: re.compile(r'\b[A-Z]+-\d+\b'),
            # User IDs (alphanumeric)
            PIIType.USER_ID: re.compile(r'\buser[_-]?\d+\b|\b[a-z]+\.\d+\b', re.IGNORECASE),
        }
    
    def _generate_anonymized_value(self, original: str, pii_type: PIIType) -> str:
        """Generate anonymized replacement value based on PII type"""
        # Use consistent hashing for same values
        hash_obj = hashlib.md5(original.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        if pii_type == PIIType.EMAIL:
            return f"user_{hash_hex}@example.com"
        elif pii_type == PIIType.PHONE:
            return f"555-{hash_hex[:3]}-{hash_hex[3:7]}"
        elif pii_type == PIIType.SSN:
            return f"XXX-XX-{hash_hex[:4]}"
        elif pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{hash_hex[:4]}"
        elif pii_type == PIIType.IP_ADDRESS:
            return f"192.168.1.{int(hash_hex[:2], 16) % 255}"
        elif pii_type == PIIType.NAME:
            return f"Person_{hash_hex[:6]}"
        elif pii_type == PIIType.TICKET_ID:
            return f"ANON-{hash_hex[:6]}"
        elif pii_type == PIIType.USER_ID:
            return f"user_{hash_hex[:8]}"
        else:
            return f"ANON_{hash_hex}"
    
    def _detect_and_anonymize_match(self, match: re.Match, pii_type: PIIType, context: str = None) -> str:
        """Process a regex match and return anonymized version"""
        original_value = match.group()
        
        # Check if we already have this value anonymized
        if original_value in self.entity_map:
            return self.entity_map[original_value].anonymized_value
        
        # Generate new anonymized value
        anonymized_value = self._generate_anonymized_value(original_value, pii_type)
        
        # Store the mapping
        entity = PIIEntity(
            original_value=original_value,
            anonymized_value=anonymized_value,
            pii_type=pii_type,
            context=context
        )
        
        self.entity_map[original_value] = entity
        self.reverse_map[anonymized_value] = original_value
        
        logger.info(f"Anonymized {pii_type.value}: {original_value} -> {anonymized_value}")
        return anonymized_value
    
    def anonymize_text(self, text: str, context: str = None) -> str:
        """Anonymize PII in text using regex patterns"""
        if not isinstance(text, str):
            return text
            
        anonymized_text = text
        
        for pii_type, pattern in self.patterns.items():
            def replace_func(match):
                return self._detect_and_anonymize_match(match, pii_type, context)
            
            anonymized_text = pattern.sub(replace_func, anonymized_text)
        
        return anonymized_text
    
    def anonymize_data_structure(self, data: Any, context: str = None) -> Any:
        """Recursively anonymize PII in complex data structures"""
        if isinstance(data, str):
            return self.anonymize_text(data, context)
        elif isinstance(data, dict):
            return {
                key: self.anonymize_data_structure(value, f"{context}.{key}" if context else key)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self.anonymize_data_structure(item, f"{context}[{i}]" if context else f"item_{i}")
                for i, item in enumerate(data)
            ]
        elif isinstance(data, tuple):
            return tuple(
                self.anonymize_data_structure(item, f"{context}[{i}]" if context else f"item_{i}")
                for i, item in enumerate(data)
            )
        else:
            return data
    
    def deanonymize_text(self, text: str) -> str:
        """Restore original PII values in anonymized text"""
        if not isinstance(text, str):
            return text
            
        deanonymized_text = text
        
        # Replace anonymized values with original ones
        for anonymized_value, original_value in self.reverse_map.items():
            deanonymized_text = deanonymized_text.replace(anonymized_value, original_value)
        
        return deanonymized_text
    
    def deanonymize_data_structure(self, data: Any) -> Any:
        """Recursively deanonymize data structures"""
        if isinstance(data, str):
            return self.deanonymize_text(data)
        elif isinstance(data, dict):
            return {key: self.deanonymize_data_structure(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.deanonymize_data_structure(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.deanonymize_data_structure(item) for item in data)
        else:
            return data
    
    def get_entity_map(self) -> Dict[str, Dict]:
        """Get serializable entity map for storage/logging"""
        return {
            original: asdict(entity) for original, entity in self.entity_map.items()
        }
    
    def clear_session_data(self):
        """Clear anonymization data for new session"""
        self.entity_map.clear()
        self.reverse_map.clear()
    
    def add_custom_pattern(self, pii_type: PIIType, pattern: str):
        """Add custom regex pattern for specific PII detection"""
        self.patterns[pii_type] = re.compile(pattern)

class AnonymizedMCPToolWrapper:
    """
    Wrapper for MCP tools that handles anonymization/deanonymization automatically
    """
    
    def __init__(self, original_tool, anonymizer: PIIAnonymizer):
        self.original_tool = original_tool
        self.anonymizer = anonymizer
        self.name = original_tool.name
        self.description = original_tool.description
        self.args_schema = original_tool.args_schema
    
    async def __call__(self, *args, **kwargs):
        """Execute tool with anonymization/deanonymization"""
        # Log original request (be careful with logging PII in production)
        logger.debug(f"Original tool call: {self.name} with args: {args}, kwargs: {kwargs}")
        
        # Don't anonymize the input to the tool - tools need real data to work
        # Instead, we'll anonymize the output from the tool before sending to LLM
        try:
            # Execute the original tool
            result = await self.original_tool(*args, **kwargs)
            
            # Anonymize the result before it goes to the LLM
            anonymized_result = self.anonymizer.anonymize_data_structure(
                result, 
                context=f"tool_{self.name}"
            )
            
            logger.debug(f"Anonymized tool result for LLM: {anonymized_result}")
            
            return anonymized_result
            
        except Exception as e:
            logger.error(f"Error in tool {self.name}: {e}")
            raise

def create_anonymized_tools(tools: List[Any], anonymizer: PIIAnonymizer) -> List[AnonymizedMCPToolWrapper]:
    """Create anonymized versions of MCP tools"""
    return [AnonymizedMCPToolWrapper(tool, anonymizer) for tool in tools]


# Modified FastAPI integration
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from tachyon_langchain_client import TachyonLangchainClient
from langgraph.prebuilt import create_react_agent
from contextlib import AsyncExitStack

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Global Variables ---
global_agent = None
global_mcp_exit_stack = None
global_anonymizer = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Anonymized MCP Langchain Agent API",
    description="API with PII anonymization for MCP and Langchain integration.",
    version="1.0.0"
)

model_client = TachyonLangchainClient(model_name="gemini-2.5-flash")

async def initialize_mcp_environment():
    """
    Sets up MCP sessions with PII anonymization
    """
    global global_agent, global_mcp_exit_stack, global_anonymizer
    
    # Initialize anonymizer
    global_anonymizer = PIIAnonymizer()
    
    server_configs = [
        ("custom_mcp_jira", StdioServerParameters(command="python", args=["-m", "custom_mcp_jira.__main__"])),
        ("jenkins_mcp", StdioServerParameters(command="python", args=["-m", "jenkins_mcp.__main__"])),
        ("KB_mcp", StdioServerParameters(command="python", args=["-m", "KB_mcp.__main__"])),
    ]
    
    all_tools_list = []
    
    # Initialize the global AsyncExitStack
    global_mcp_exit_stack = AsyncExitStack()
    await global_mcp_exit_stack.__aenter__()
    
    for name, params in server_configs:
        print(f"Loading tools from {name}...")
        
        read_pipe, write_pipe = await global_mcp_exit_stack.enter_async_context(stdio_client(params))
        session = await global_mcp_exit_stack.enter_async_context(ClientSession(read_pipe, write_pipe))
        await session.initialize()
        
        tools = await load_mcp_tools(session)
        
        # Wrap tools with anonymization
        anonymized_tools = create_anonymized_tools(tools, global_anonymizer)
        all_tools_list.extend(anonymized_tools)
        
        print(f"Loaded {len(tools)} anonymized tools from {name}.")
    
    print(f"Total anonymized tools loaded: {len(all_tools_list)}")
    global_agent = create_react_agent(model_client, all_tools_list)
    print("Anonymized agent initialized successfully.")

@app.on_event("startup")
async def startup_event():
    print("FastAPI startup event triggered. Initializing anonymized MCP environment...")
    await initialize_mcp_environment()

@app.on_event("shutdown")
async def shutdown_event():
    print("FastAPI shutdown event triggered. Cleaning up...")
    global global_mcp_exit_stack
    if global_mcp_exit_stack:
        await global_mcp_exit_stack.__aexit__(None, None, None)
    print("MCP environment cleaned up.")

class AgentInput(BaseModel):
    message: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: Any
    anonymization_stats: Dict[str, int]

@app.post("/invoke_agent", response_model=AgentResponse)
async def invoke_agent_endpoint(input_data: AgentInput):
    """
    Invoke agent with PII anonymization/deanonymization
    """
    global global_agent, global_anonymizer
    
    if global_agent is None or global_anonymizer is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please wait for application startup."
        )
    
    try:
        # Clear previous session data if new session
        if input_data.session_id:
            global_anonymizer.clear_session_data()
        
        # Anonymize input message before sending to LLM
        anonymized_message = global_anonymizer.anonymize_text(
            input_data.message, 
            context="user_input"
        )
        
        print(f"Anonymized user message: {anonymized_message}")
        
        # Invoke agent with anonymized message
        # The agent will receive anonymized data from tools automatically
        agent_response = await global_agent.ainvoke({
            "messages": anonymized_message
        })
        
        print(f"Agent response (anonymized): {agent_response}")
        
        # Deanonymize the response before returning to user
        deanonymized_response = global_anonymizer.deanonymize_data_structure(agent_response)
        
        # Get anonymization statistics
        entity_map = global_anonymizer.get_entity_map()
        anonymization_stats = {
            pii_type.value: sum(1 for entity in entity_map.values() 
                              if entity['pii_type'] == pii_type.value)
            for pii_type in PIIType
        }
        
        return AgentResponse(
            response=deanonymized_response,
            anonymization_stats=anonymization_stats
        )
        
    except Exception as e:
        logger.error(f"Error in anonymized agent call: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/anonymization_stats")
async def get_anonymization_stats():
    """Get current anonymization statistics"""
    global global_anonymizer
    
    if global_anonymizer is None:
        raise HTTPException(status_code=503, detail="Anonymizer not initialized")
    
    entity_map = global_anonymizer.get_entity_map()
    
    stats = {
        "total_entities": len(entity_map),
        "by_type": {
            pii_type.value: sum(1 for entity in entity_map.values() 
                              if entity['pii_type'] == pii_type.value)
            for pii_type in PIIType
        },
        "entities": list(entity_map.values())  # Be careful with this in production
    }
    
    return stats

if __name__ == "__main__":
    uvicorn.run("anonymized_mcp_agent:app", host="0.0.0.0", port=8000, reload=True)
