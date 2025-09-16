import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional

# MCP and Langchain imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from tachyon_langchain_client import TachyonLangchainClient
from langgraph.prebuilt import create_react_agent

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Import the PII anonymization system
from your_anonymization_module import PIIAnonymizer, AnonymizedMCPToolWrapper, create_anonymized_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
global_agent = None
global_mcp_exit_stack = None
global_anonymizer = None
global_mcp_sessions = []

app = FastAPI(
    title="Fixed Anonymized MCP Langchain Agent API",
    description="Fixed version with proper async context management",
    version="1.0.1"
)

model_client = TachyonLangchainClient(model_name="gemini-2.5-flash")

class MCPServerManager:
    """Manages MCP server connections with proper error handling"""
    
    def __init__(self):
        self.sessions = []
        self.exit_stack = None
        self.tools = []
    
    async def initialize_server(self, name: str, params: StdioServerParameters) -> List[Any]:
        """Initialize a single MCP server with error handling"""
        try:
            logger.info(f"Initializing MCP server: {name}")
            
            # Create stdio client with timeout
            read_pipe, write_pipe = await asyncio.wait_for(
                stdio_client(params),
                timeout=30.0  # 30 second timeout
            )
            
            # Create and initialize session
            session = ClientSession(read_pipe, write_pipe)
            await asyncio.wait_for(
                session.initialize(),
                timeout=10.0  # 10 second timeout for initialization
            )
            
            # Load tools from this server
            tools = await asyncio.wait_for(
                load_mcp_tools(session),
                timeout=15.0  # 15 second timeout for loading tools
            )
            
            # Store session for cleanup
            self.sessions.append({
                'name': name,
                'session': session,
                'read_pipe': read_pipe,
                'write_pipe': write_pipe,
                'tools': tools
            })
            
            logger.info(f"Successfully loaded {len(tools)} tools from {name}")
            return tools
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout initializing MCP server: {name}")
            return []
        except Exception as e:
            logger.error(f"Error initializing MCP server {name}: {e}")
            return []
    
    async def initialize_all_servers(self) -> List[Any]:
        """Initialize all MCP servers concurrently with error handling"""
        server_configs = [
            ("custom_mcp_jira", StdioServerParameters(
                command="python", 
                args=["-m", "custom_mcp_jira.__main__"]
            )),
            ("jenkins_mcp", StdioServerParameters(
                command="python", 
                args=["-m", "jenkins_mcp.__main__"]
            )),
            ("KB_mcp", StdioServerParameters(
                command="python", 
                args=["-m", "KB_mcp.__main__"]
            )),
        ]
        
        all_tools = []
        
        # Initialize servers one by one to avoid async context conflicts
        for name, params in server_configs:
            try:
                tools = await self.initialize_server(name, params)
                all_tools.extend(tools)
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                # Continue with other servers even if one fails
                continue
        
        logger.info(f"Total tools loaded from all servers: {len(all_tools)}")
        return all_tools
    
    async def cleanup(self):
        """Clean up all MCP sessions"""
        logger.info("Cleaning up MCP sessions...")
        
        for session_info in self.sessions:
            try:
                name = session_info['name']
                session = session_info['session']
                
                logger.info(f"Closing session for {name}")
                
                # Close session gracefully
                if hasattr(session, 'close'):
                    await session.close()
                
                # Close pipes
                read_pipe = session_info.get('read_pipe')
                write_pipe = session_info.get('write_pipe')
                
                if read_pipe and hasattr(read_pipe, 'close'):
                    await read_pipe.close()
                if write_pipe and hasattr(write_pipe, 'close'):
                    await write_pipe.close()
                    
            except Exception as e:
                logger.error(f"Error cleaning up session {session_info['name']}: {e}")
        
        self.sessions.clear()
        logger.info("MCP cleanup completed")

# Global MCP manager
mcp_manager = MCPServerManager()

async def initialize_mcp_environment():
    """Initialize MCP environment with improved error handling"""
    global global_agent, global_anonymizer
    
    try:
        logger.info("Starting MCP environment initialization...")
        
        # Initialize anonymizer
        global_anonymizer = PIIAnonymizer()
        logger.info("PII Anonymizer initialized")
        
        # Initialize all MCP servers
        all_tools = await mcp_manager.initialize_all_servers()
        
        if not all_tools:
            logger.warning("No tools were loaded from any MCP server")
            # You might want to continue with a limited agent or raise an exception
            raise HTTPException(
                status_code=503, 
                detail="No MCP tools available - check server configurations"
            )
        
        # Create anonymized tools
        anonymized_tools = create_anonymized_tools(all_tools, global_anonymizer)
        logger.info(f"Created {len(anonymized_tools)} anonymized tool wrappers")
        
        # Create the agent
        global_agent = create_react_agent(model_client, anonymized_tools)
        logger.info("Langchain ReAct agent created successfully")
        
        logger.info("MCP environment initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP environment: {e}")
        # Clean up any partially initialized resources
        await mcp_manager.cleanup()
        raise

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event with better error handling"""
    try:
        logger.info("FastAPI startup event triggered")
        await initialize_mcp_environment()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        # You might want to exit the application here
        # import sys
        # sys.exit(1)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event"""
    try:
        logger.info("FastAPI shutdown event triggered")
        await mcp_manager.cleanup()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

class AgentInput(BaseModel):
    message: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: Any
    anonymization_stats: Dict[str, int]
    status: str = "success"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global global_agent, global_anonymizer
    
    status = {
        "status": "healthy" if global_agent and global_anonymizer else "initializing",
        "agent_ready": global_agent is not None,
        "anonymizer_ready": global_anonymizer is not None,
        "active_sessions": len(mcp_manager.sessions),
        "available_tools": len(mcp_manager.sessions)
    }
    
    if not global_agent:
        status["status"] = "unhealthy"
        status["message"] = "Agent not initialized"
    
    return status

@app.post("/invoke_agent", response_model=AgentResponse)
async def invoke_agent_endpoint(input_data: AgentInput):
    """Invoke agent with improved error handling"""
    global global_agent, global_anonymizer
    
    # Check if services are ready
    if global_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check /health endpoint for status."
        )
    
    if global_anonymizer is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymizer not initialized. Check /health endpoint for status."
        )
    
    try:
        # Clear previous session data if new session
        if input_data.session_id:
            global_anonymizer.clear_session_data()
            logger.info(f"Cleared anonymization data for session: {input_data.session_id}")
        
        # Anonymize input message
        anonymized_message = global_anonymizer.anonymize_text(
            input_data.message, 
            context="user_input"
        )
        
        logger.info(f"Processing request - Original length: {len(input_data.message)}, "
                   f"Anonymized length: {len(anonymized_message)}")
        
        # Invoke agent with timeout
        agent_response = await asyncio.wait_for(
            global_agent.ainvoke({"messages": anonymized_message}),
            timeout=120.0  # 2 minute timeout for agent processing
        )
        
        logger.info("Agent processing completed")
        
        # Deanonymize the response
        deanonymized_response = global_anonymizer.deanonymize_data_structure(agent_response)
        
        # Get anonymization statistics
        entity_map = global_anonymizer.get_entity_map()
        anonymization_stats = {
            "total_entities": len(entity_map),
            "emails_anonymized": sum(1 for e in entity_map.values() if e.pii_type.value == "email"),
            "phones_anonymized": sum(1 for e in entity_map.values() if e.pii_type.value == "phone"),
            "tickets_anonymized": sum(1 for e in entity_map.values() if e.pii_type.value == "ticket_id"),
            "names_anonymized": sum(1 for e in entity_map.values() if e.pii_type.value == "name"),
        }
        
        return AgentResponse(
            response=deanonymized_response,
            anonymization_stats=anonymization_stats,
            status="success"
        )
        
    except asyncio.TimeoutError:
        logger.error("Agent processing timeout")
        raise HTTPException(
            status_code=504,
            detail="Agent processing timeout. Please try again with a simpler request."
        )
    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing error: {str(e)}"
        )

@app.get("/anonymization_stats")
async def get_anonymization_stats():
    """Get current anonymization statistics"""
    global global_anonymizer
    
    if global_anonymizer is None:
        raise HTTPException(status_code=503, detail="Anonymizer not initialized")
    
    try:
        entity_map = global_anonymizer.get_entity_map()
        
        stats = {
            "total_entities": len(entity_map),
            "by_type": {},
            "session_active": len(entity_map) > 0
        }
        
        # Count by PII type
        for entity in entity_map.values():
            pii_type = entity['pii_type']
            stats["by_type"][pii_type] = stats["by_type"].get(pii_type, 0) + 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting anonymization stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )

@app.post("/clear_session")
async def clear_session():
    """Clear current anonymization session"""
    global global_anonymizer
    
    if global_anonymizer is None:
        raise HTTPException(status_code=503, detail="Anonymizer not initialized")
    
    try:
        entities_cleared = len(global_anonymizer.entity_map)
        global_anonymizer.clear_session_data()
        
        return {
            "status": "success",
            "entities_cleared": entities_cleared,
            "message": f"Cleared {entities_cleared} anonymized entities"
        }
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "fixed_mcp_agent:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload to avoid async context issues
        log_level="info"
    )
