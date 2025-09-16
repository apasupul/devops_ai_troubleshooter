import asyncio
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

# Presidio imports
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

# MCP and Langchain imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from tachyon_langchain_client import TachyonLangchainClient
from langgraph.prebuilt import create_react_agent

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# ENHANCED PII ANONYMIZATION WITH PRESIDIO
# ==============================================================================

class PIIType(Enum):
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"
    URL = "URL"
    US_SSN = "US_SSN"
    # Custom patterns for your domain
    JIRA_TICKET = "JIRA_TICKET"
    JENKINS_JOB = "JENKINS_JOB"
    USER_ID = "USER_ID"
    SERVER_NAME = "SERVER_NAME"

@dataclass
class PIIEntity:
    original_value: str
    anonymized_value: str
    pii_type: str
    confidence_score: float
    start: int
    end: int
    context: Optional[str] = None

class PresidioPIIAnonymizer:
    """
    Enhanced PII anonymization using Microsoft Presidio for better accuracy
    """
    
    def __init__(self, supported_languages: List[str] = None):
        self.supported_languages = supported_languages or ["en"]
        self.entity_map: Dict[str, PIIEntity] = {}
        self.reverse_map: Dict[str, str] = {}
        
        # Initialize Presidio engines
        self._initialize_presidio_engines()
        
        # Add custom recognizers
        self._add_custom_recognizers()
        
    def _initialize_presidio_engines(self):
        """Initialize Presidio analyzer and anonymizer engines"""
        try:
            # Configure NLP engine (using spaCy)
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            
            # Create NLP engine provider
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            
            # Initialize analyzer
            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=self.supported_languages
            )
            
            # Initialize anonymizer
            self.anonymizer = AnonymizerEngine()
            
            logger.info("Presidio engines initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Presidio with spaCy: {e}")
            logger.info("Falling back to basic Presidio configuration")
            
            # Fallback to basic configuration
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
    
    def _add_custom_recognizers(self):
        """Add custom recognizers for domain-specific PII"""
        
        # JIRA Ticket recognizer
        jira_pattern = Pattern(
            name="jira_ticket_pattern",
            regex=r"\b[A-Z]{2,10}-\d+\b",
            score=0.8
        )
        jira_recognizer = PatternRecognizer(
            supported_entity="JIRA_TICKET",
            patterns=[jira_pattern]
        )
        
        # Jenkins Job recognizer
        jenkins_pattern = Pattern(
            name="jenkins_job_pattern", 
            regex=r"\bjob/[\w\-_/]+\b|\bbuild/\d+\b",
            score=0.7
        )
        jenkins_recognizer = PatternRecognizer(
            supported_entity="JENKINS_JOB",
            patterns=[jenkins_pattern]
        )
        
        # User ID recognizer
        user_id_pattern = Pattern(
            name="user_id_pattern",
            regex=r"\buser[_\-]?\d+\b|\b[a-z]+\.[a-z]+\.\d+\b",
            score=0.6
        )
        user_id_recognizer = PatternRecognizer(
            supported_entity="USER_ID",
            patterns=[user_id_pattern]
        )
        
        # Server name recognizer
        server_pattern = Pattern(
            name="server_name_pattern",
            regex=r"\b(?:server|host|node)[_\-]?[\w\d]+\b|\b[\w\d]+-(?:prod|dev|test|staging)\b",
            score=0.7
        )
        server_recognizer = PatternRecognizer(
            supported_entity="SERVER_NAME",
            patterns=[server_pattern]
        )
        
        # Add all custom recognizers
        self.analyzer.registry.add_recognizer(jira_recognizer)
        self.analyzer.registry.add_recognizer(jenkins_recognizer) 
        self.analyzer.registry.add_recognizer(user_id_recognizer)
        self.analyzer.registry.add_recognizer(server_recognizer)
        
        logger.info("Added custom PII recognizers for JIRA, Jenkins, User IDs, and Server names")
    
    def _generate_anonymized_value(self, original: str, pii_type: str) -> str:
        """Generate consistent anonymized replacement based on PII type"""
        # Use consistent hashing for same values
        hash_obj = hashlib.md5(original.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        
        anonymization_map = {
            "EMAIL_ADDRESS": f"user_{hash_hex}@example.com",
            "PHONE_NUMBER": f"555-{hash_hex[:3]}-{hash_hex[3:7]}",
            "PERSON": f"Person_{hash_hex[:6]}",
            "LOCATION": f"Location_{hash_hex[:6]}", 
            "ORGANIZATION": f"Company_{hash_hex[:6]}",
            "CREDIT_CARD": f"****-****-****-{hash_hex[:4]}",
            "IP_ADDRESS": f"192.168.1.{int(hash_hex[:2], 16) % 255}",
            "US_SSN": f"XXX-XX-{hash_hex[:4]}",
            "DATE_TIME": f"REDACTED_DATE_{hash_hex[:6]}",
            "URL": f"https://example.com/{hash_hex[:8]}",
            # Custom types
            "JIRA_TICKET": f"ANON-{hash_hex[:6]}",
            "JENKINS_JOB": f"job/anon_{hash_hex[:8]}",
            "USER_ID": f"user_{hash_hex[:8]}",
            "SERVER_NAME": f"server_{hash_hex[:6]}",
        }
        
        return anonymization_map.get(pii_type, f"ANON_{pii_type}_{hash_hex}")
    
    def analyze_text(self, text: str, context: str = None) -> List[RecognizerResult]:
        """Analyze text for PII using Presidio"""
        try:
            # Get all PII entities from Presidio
            results = self.analyzer.analyze(
                text=text,
                language="en",
                score_threshold=0.5  # Minimum confidence score
            )
            
            logger.debug(f"Found {len(results)} PII entities in text")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing text with Presidio: {e}")
            return []
    
    def anonymize_text_with_presidio(self, text: str, context: str = None) -> str:
        """Anonymize text using Presidio's built-in anonymizer"""
        try:
            # Analyze text for PII
            analyzer_results = self.analyze_text(text, context)
            
            if not analyzer_results:
                return text
                
            # Create operator configs for anonymization
            operators = {}
            for result in analyzer_results:
                entity_type = result.entity_type
                original_value = text[result.start:result.end]
                
                # Check if we already have this value anonymized
                if original_value in self.entity_map:
                    anonymized_value = self.entity_map[original_value].anonymized_value
                else:
                    # Generate new anonymized value
                    anonymized_value = self._generate_anonymized_value(original_value, entity_type)
                    
                    # Store mapping
                    entity = PIIEntity(
                        original_value=original_value,
                        anonymized_value=anonymized_value,
                        pii_type=entity_type,
                        confidence_score=result.score,
                        start=result.start,
                        end=result.end,
                        context=context
                    )
                    
                    self.entity_map[original_value] = entity
                    self.reverse_map[anonymized_value] = original_value
                
                # Set up operator for this entity type to use our custom value
                operators[entity_type] = OperatorConfig(
                    "replace",
                    {"new_value": anonymized_value}
                )
            
            # Anonymize using Presidio
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            
            return anonymized_result.text
            
        except Exception as e:
            logger.error(f"Error anonymizing text with Presidio: {e}")
            return text
    
    def anonymize_text(self, text: str, context: str = None) -> str:
        """Main method to anonymize text"""
        if not isinstance(text, str) or not text.strip():
            return text
            
        return self.anonymize_text_with_presidio(text, context)
    
    def anonymize_data_structure(self, data: Any, context: str = None) -> Any:
        """Recursively anonymize PII in complex data structures"""
        if isinstance(data, str):
            return self.anonymize_text(data, context)
        elif isinstance(data, dict):
            return {
                key: self.anonymize_data_structure(
                    value, 
                    f"{context}.{key}" if context else key
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self.anonymize_data_structure(
                    item, 
                    f"{context}[{i}]" if context else f"item_{i}"
                )
                for i, item in enumerate(data)
            ]
        elif isinstance(data, tuple):
            return tuple(
                self.anonymize_data_structure(
                    item, 
                    f"{context}[{i}]" if context else f"item_{i}"
                )
                for i, item in enumerate(data)
            )
        else:
            return data
    
    def deanonymize_text(self, text: str) -> str:
        """Restore original PII values in anonymized text"""
        if not isinstance(text, str):
            return text
            
        deanonymized_text = text
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_items = sorted(
            self.reverse_map.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for anonymized_value, original_value in sorted_items:
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
        """Get serializable entity map"""
        return {
            original: asdict(entity) for original, entity in self.entity_map.items()
        }
    
    def get_anonymization_stats(self) -> Dict[str, Any]:
        """Get detailed anonymization statistics"""
        stats = {
            "total_entities": len(self.entity_map),
            "by_type": {},
            "by_confidence": {"high": 0, "medium": 0, "low": 0},
            "entities_by_context": {}
        }
        
        for entity in self.entity_map.values():
            # Count by type
            pii_type = entity.pii_type
            stats["by_type"][pii_type] = stats["by_type"].get(pii_type, 0) + 1
            
            # Count by confidence
            if entity.confidence_score >= 0.8:
                stats["by_confidence"]["high"] += 1
            elif entity.confidence_score >= 0.6:
                stats["by_confidence"]["medium"] += 1
            else:
                stats["by_confidence"]["low"] += 1
            
            # Count by context
            if entity.context:
                stats["entities_by_context"][entity.context] = stats["entities_by_context"].get(entity.context, 0) + 1
        
        return stats
    
    def clear_session_data(self):
        """Clear anonymization data for new session"""
        self.entity_map.clear()
        self.reverse_map.clear()
        logger.info("Cleared anonymization session data")

class AnonymizedMCPToolWrapper:
    """
    Enhanced wrapper using Presidio for PII detection
    """
    
    def __init__(self, original_tool, anonymizer: PresidioPIIAnonymizer):
        self.original_tool = original_tool
        self.anonymizer = anonymizer
        self.name = original_tool.name
        self.description = original_tool.description
        self.args_schema = original_tool.args_schema
    
    async def __call__(self, *args, **kwargs):
        """Execute tool with Presidio-based anonymization"""
        logger.debug(f"Tool {self.name} called with Presidio anonymization")
        
        try:
            # Execute the original tool
            result = await self.original_tool(*args, **kwargs)
            
            # Anonymize the result using Presidio
            anonymized_result = self.anonymizer.anonymize_data_structure(
                result, 
                context=f"tool_{self.name}"
            )
            
            logger.debug(f"Tool {self.name} result anonymized with Presidio")
            return anonymized_result
            
        except Exception as e:
            logger.error(f"Error in tool {self.name}: {e}")
            raise

def create_anonymized_tools(tools: List[Any], anonymizer: PresidioPIIAnonymizer) -> List[AnonymizedMCPToolWrapper]:
    """Create Presidio-enhanced anonymized tools"""
    return [AnonymizedMCPToolWrapper(tool, anonymizer) for tool in tools]

# ==============================================================================
# MCP SERVER MANAGER (Same as before)
# ==============================================================================

class MCPServerManager:
    """Manages MCP server connections"""
    
    def __init__(self):
        self.sessions = []
    
    async def initialize_server(self, name: str, params: StdioServerParameters) -> List[Any]:
        """Initialize a single MCP server"""
        try:
            logger.info(f"Initializing MCP server: {name}")
            
            read_pipe, write_pipe = await asyncio.wait_for(
                stdio_client(params),
                timeout=30.0
            )
            
            session = ClientSession(read_pipe, write_pipe)
            await asyncio.wait_for(
                session.initialize(),
                timeout=10.0
            )
            
            tools = await asyncio.wait_for(
                load_mcp_tools(session),
                timeout=15.0
            )
            
            self.sessions.append({
                'name': name,
                'session': session,
                'read_pipe': read_pipe,
                'write_pipe': write_pipe,
                'tools': tools
            })
            
            logger.info(f"Successfully loaded {len(tools)} tools from {name}")
            return tools
            
        except Exception as e:
            logger.error(f"Error initializing MCP server {name}: {e}")
            return []
    
    async def initialize_all_servers(self) -> List[Any]:
        """Initialize all MCP servers"""
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
        for name, params in server_configs:
            tools = await self.initialize_server(name, params)
            all_tools.extend(tools)
        
        logger.info(f"Total tools loaded: {len(all_tools)}")
        return all_tools
    
    async def cleanup(self):
        """Clean up MCP sessions"""
        logger.info("Cleaning up MCP sessions...")
        
        for session_info in self.sessions:
            try:
                name = session_info['name']
                session = session_info['session']
                
                if hasattr(session, 'close'):
                    await session.close()
                
                read_pipe = session_info.get('read_pipe')
                write_pipe = session_info.get('write_pipe')
                
                if read_pipe and hasattr(read_pipe, 'close'):
                    await read_pipe.close()
                if write_pipe and hasattr(write_pipe, 'close'):
                    await write_pipe.close()
                    
            except Exception as e:
                logger.error(f"Error cleaning up {session_info['name']}: {e}")
        
        self.sessions.clear()

# ==============================================================================
# FASTAPI APPLICATION WITH PRESIDIO
# ==============================================================================

global_agent = None
global_anonymizer = None
mcp_manager = MCPServerManager()

app = FastAPI(
    title="MCP Agent with Presidio PII Protection",
    description="Enhanced PII anonymization using Microsoft Presidio",
    version="2.0.0"
)

model_client = TachyonLangchainClient(model_name="gemini-2.5-flash")

async def initialize_mcp_environment():
    """Initialize MCP environment with Presidio anonymization"""
    global global_agent, global_anonymizer
    
    try:
        logger.info("Initializing Presidio-enhanced MCP environment...")
        
        # Initialize Presidio anonymizer
        global_anonymizer = PresidioPIIAnonymizer()
        logger.info("Presidio PII Anonymizer initialized")
        
        # Initialize MCP servers
        all_tools = await mcp_manager.initialize_all_servers()
        
        if not all_tools:
            raise Exception("No MCP tools available")
        
        # Create Presidio-enhanced anonymized tools
        anonymized_tools = create_anonymized_tools(all_tools, global_anonymizer)
        logger.info(f"Created {len(anonymized_tools)} Presidio-enhanced tool wrappers")
        
        # Create agent
        global_agent = create_react_agent(model_client, anonymized_tools)
        logger.info("Presidio-enhanced agent created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Presidio environment: {e}")
        await mcp_manager.cleanup()
        raise

@app.on_event("startup")
async def startup_event():
    """FastAPI startup with Presidio"""
    try:
        logger.info("Starting Presidio-enhanced application...")
        await initialize_mcp_environment()
        logger.info("Presidio application startup completed")
    except Exception as e:
        logger.error(f"Presidio application startup failed: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """FastAPI shutdown"""
    try:
        await mcp_manager.cleanup()
        logger.info("Presidio application shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

class AgentInput(BaseModel):
    message: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: Any
    anonymization_stats: Dict[str, Any]
    presidio_analysis: Dict[str, Any]
    status: str = "success"

@app.get("/health")
async def health_check():
    """Health check with Presidio status"""
    global global_agent, global_anonymizer
    
    presidio_status = "unknown"
    if global_anonymizer:
        try:
            # Test Presidio with a simple analysis
            test_results = global_anonymizer.analyze_text("test@example.com")
            presidio_status = "working" if test_results else "initialized"
        except Exception as e:
            presidio_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if global_agent and global_anonymizer else "initializing",
        "agent_ready": global_agent is not None,
        "presidio_ready": global_anonymizer is not None,
        "presidio_status": presidio_status,
        "active_sessions": len(mcp_manager.sessions)
    }

@app.post("/invoke_agent", response_model=AgentResponse)
async def invoke_agent_endpoint(input_data: AgentInput):
    """Invoke agent with Presidio PII protection"""
    global global_agent, global_anonymizer
    
    if not global_agent or not global_anonymizer:
        raise HTTPException(
            status_code=503,
            detail="Presidio services not ready"
        )
    
    try:
        if input_data.session_id:
            global_anonymizer.clear_session_data()
        
        # Analyze original message for PII insights
        original_analysis = global_anonymizer.analyze_text(input_data.message, "user_input")
        
        # Anonymize input using Presidio
        anonymized_message = global_anonymizer.anonymize_text(
            input_data.message, 
            context="user_input"
        )
        
        logger.info(f"Presidio found {len(original_analysis)} PII entities")
        
        # Invoke agent
        agent_response = await asyncio.wait_for(
            global_agent.ainvoke({"messages": anonymized_message}),
            timeout=120.0
        )
        
        # Deanonymize response
        deanonymized_response = global_anonymizer.deanonymize_data_structure(agent_response)
        
        # Get comprehensive stats
        anonymization_stats = global_anonymizer.get_anonymization_stats()
        
        # Presidio analysis summary
        presidio_analysis = {
            "entities_found": len(original_analysis),
            "entity_types": list(set(result.entity_type for result in original_analysis)),
            "high_confidence_entities": len([r for r in original_analysis if r.score >= 0.8]),
            "custom_patterns_matched": len([r for r in original_analysis if r.entity_type in 
                                          ["JIRA_TICKET", "JENKINS_JOB", "USER_ID", "SERVER_NAME"]])
        }
        
        return AgentResponse(
            response=deanonymized_response,
            anonymization_stats=anonymization_stats,
            presidio_analysis=presidio_analysis,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Presidio agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/presidio_stats")
async def get_presidio_stats():
    """Get detailed Presidio anonymization statistics"""
    global global_anonymizer
    
    if not global_anonymizer:
        raise HTTPException(status_code=503, detail="Presidio not initialized")
    
    return global_anonymizer.get_anonymization_stats()

if __name__ == "__main__":
    uvicorn.run(
        "presidio_app:app",  # Replace with your filename
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
