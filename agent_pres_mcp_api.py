#!/usr/bin/env python3
"""
Presidio Dual Server with FastAPI and FastMCP
Provides secure anonymization and de-anonymization tools with vault storage
Available as both REST API endpoints and MCP tools
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# FastMCP and MCP imports
from fastmcp import FastMCP
from mcp.types import Tool, TextContent

# Presidio imports
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for FastAPI
class AnalyzePIIRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for PII")
    language: str = Field(default="en", description="Language code")
    entities: Optional[List[str]] = Field(default=None, description="Specific entities to detect")

class CreateVaultRequest(BaseModel):
    metadata: Optional[Dict] = Field(default=None, description="Optional metadata for the vault entry")

class AnonymizeRequest(BaseModel):
    text: str = Field(..., description="Text to anonymize")
    language: str = Field(default="en", description="Language code")
    entities: Optional[List[str]] = Field(default=None, description="Specific entities to anonymize")
    anonymization_method: str = Field(default="replace", description="Anonymization method")
    vault_id: Optional[str] = Field(default=None, description="Existing vault ID to store result in")
    store_in_vault: bool = Field(default=True, description="Store result in vault")

class VaultRetrieveRequest(BaseModel):
    vault_id: str = Field(..., description="Vault ID to retrieve")

class CustomRecognizerRequest(BaseModel):
    recognizer_config: Dict = Field(..., description="Custom recognizer configuration")

class SecureVault:
    """Secure vault for storing anonymized data and mapping keys"""
    
    def __init__(self, vault_path: str = "presidio_vault.db", password: str = None):
        self.vault_path = vault_path
        self.password = password or os.environ.get("VAULT_PASSWORD", "default_password")
        self.key = self._derive_key(self.password)
        self.fernet = Fernet(self.key)
        self._init_database()
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        password_bytes = password.encode()
        salt = b'presidio_salt'  # In production, use a random salt per vault
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _init_database(self):
        """Initialize the secure database"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        # Create tables for anonymized data and mappings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anonymized_data (
                id TEXT PRIMARY KEY,
                original_hash TEXT,
                anonymized_data BLOB,
                metadata BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_mappings (
                id TEXT PRIMARY KEY,
                data_id TEXT,
                original_entity_hash TEXT,
                anonymized_value TEXT,
                entity_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (data_id) REFERENCES anonymized_data (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_data(self, data: str) -> str:
        """Create hash of data for indexing"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def create_vault_entry(self, metadata: Dict = None) -> str:
        """Create a new vault entry and return the vault_id
        
        Args:
            metadata: Optional metadata for the vault entry
            
        Returns:
            str: The generated vault_id
        """
        vault_id = str(uuid.uuid4())
        
        # Create a placeholder entry in the database
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        try:
            # Encrypt empty placeholder data and metadata
            placeholder_data = self.fernet.encrypt("".encode())
            encrypted_metadata = self.fernet.encrypt(json.dumps(metadata or {"status": "created", "data_stored": False}).encode())
            
            cursor.execute('''
                INSERT INTO anonymized_data (id, original_hash, anonymized_data, metadata)
                VALUES (?, ?, ?, ?)
            ''', (vault_id, "", placeholder_data, encrypted_metadata))
            
            conn.commit()
            return vault_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def update_vault_entry(self, vault_id: str, original_data: str, anonymized_data: str, 
                          entity_mappings: Dict[str, str], metadata: Dict = None) -> bool:
        """Update an existing vault entry with anonymized data
        
        Args:
            vault_id: Existing vault ID
            original_data: Original text data
            anonymized_data: Anonymized text data
            entity_mappings: Mapping of original to anonymized entities
            metadata: Additional metadata to merge
            
        Returns:
            bool: True if successful, False if vault_id doesn't exist
        """
        # Check if vault entry exists
        existing_data = self.retrieve_anonymized_data(vault_id)
        if not existing_data:
            return False
        
        original_hash = self._hash_data(original_data)
        
        # Merge metadata
        existing_metadata = existing_data.get("metadata", {})
        if metadata:
            existing_metadata.update(metadata)
        existing_metadata["data_stored"] = True
        existing_metadata["updated_at"] = datetime.now().isoformat()
        
        # Encrypt the new data
        encrypted_data = self.fernet.encrypt(anonymized_data.encode())
        encrypted_metadata = self.fernet.encrypt(json.dumps(existing_metadata).encode())
        
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        try:
            # Update the main record
            cursor.execute('''
                UPDATE anonymized_data 
                SET original_hash = ?, anonymized_data = ?, metadata = ?
                WHERE id = ?
            ''', (original_hash, encrypted_data, encrypted_metadata, vault_id))
            
            # Delete existing entity mappings for this vault_id
            cursor.execute('DELETE FROM entity_mappings WHERE data_id = ?', (vault_id,))
            
            # Store new entity mappings
            for original_entity, anonymized_value in entity_mappings.items():
                entity_hash = self._hash_data(original_entity)
                mapping_id = str(uuid.uuid4())
                
                cursor.execute('''
                    INSERT INTO entity_mappings 
                    (id, data_id, original_entity_hash, anonymized_value, entity_type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (mapping_id, vault_id, entity_hash, anonymized_value, "UNKNOWN"))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def retrieve_anonymized_data(self, data_id: str) -> Optional[Dict]:
        """Retrieve anonymized data from vault"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT anonymized_data, metadata, created_at 
            FROM anonymized_data WHERE id = ?
        ''', (data_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            encrypted_data, encrypted_metadata, created_at = result
            decrypted_data = self.fernet.decrypt(encrypted_data).decode()
            decrypted_metadata = json.loads(self.fernet.decrypt(encrypted_metadata).decode())
            
            return {
                "id": data_id,
                "anonymized_data": decrypted_data,
                "metadata": decrypted_metadata,
                "created_at": created_at
            }
        
        return None
    
    def get_entity_mappings(self, data_id: str) -> Dict[str, str]:
        """Get entity mappings for de-anonymization"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT anonymized_value, original_entity_hash 
            FROM entity_mappings WHERE data_id = ?
        ''', (data_id,))
        
        mappings = {}
        for row in cursor.fetchall():
            anonymized_value, original_hash = row
            # Note: We store hashes, so we can't directly reverse them
            # This is a limitation for security - we'd need to store encrypted originals
            mappings[anonymized_value] = f"ORIGINAL_HASH_{original_hash[:8]}"
        
    def store_anonymized_data(self, original_data: str, anonymized_data: str, 
                            entity_mappings: Dict[str, str], metadata: Dict = None, 
                            vault_id: str = None) -> str:
        """Store anonymized data and entity mappings in vault
        
        Args:
            original_data: Original text data
            anonymized_data: Anonymized text data  
            entity_mappings: Mapping of original to anonymized entities
            metadata: Additional metadata
            vault_id: Optional existing vault_id to update, creates new if None
            
        Returns:
            str: The vault_id (existing or newly created)
        """
        if vault_id:
            # Update existing vault entry
            success = self.update_vault_entry(vault_id, original_data, anonymized_data, entity_mappings, metadata)
            if success:
                return vault_id
            else:
                raise ValueError(f"Vault ID {vault_id} does not exist")
        else:
            # Create new vault entry
            new_vault_id = str(uuid.uuid4())
            original_hash = self._hash_data(original_data)
            
            # Encrypt the anonymized data and metadata
            encrypted_data = self.fernet.encrypt(anonymized_data.encode())
            encrypted_metadata = self.fernet.encrypt(json.dumps(metadata or {}).encode())
            
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            
            try:
                # Store main anonymized data record
                cursor.execute('''
                    INSERT INTO anonymized_data (id, original_hash, anonymized_data, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (new_vault_id, original_hash, encrypted_data, encrypted_metadata))
                
                # Store entity mappings
                for original_entity, anonymized_value in entity_mappings.items():
                    entity_hash = self._hash_data(original_entity)
                    mapping_id = str(uuid.uuid4())
                    
                    cursor.execute('''
                        INSERT INTO entity_mappings 
                        (id, data_id, original_entity_hash, anonymized_value, entity_type)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (mapping_id, new_vault_id, entity_hash, anonymized_value, "UNKNOWN"))
                
                conn.commit()
                return new_vault_id
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def list_entries(self, limit: int = 10) -> List[Dict]:
        """List vault entries"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, original_hash, created_at 
            FROM anonymized_data 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        entries = []
        for row in cursor.fetchall():
            vault_id, original_hash, created_at = row
            entries.append({
                "vault_id": vault_id,
                "original_hash": original_hash[:16] + "...",
                "created_at": created_at
            })
        
        conn.close()
        return entries


class PresidioService:
    """Core Presidio service class used by both FastAPI and FastMCP"""
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.vault = SecureVault()
    
    def analyze_pii(self, text: str, language: str = "en", entities: Optional[List[str]] = None) -> Dict:
        """Analyze text for PII entities"""
        try:
            results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=entities
            )
            
            entities_found = []
            for result in results:
                entities_found.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                })
            
            return {
                "status": "success",
                "entities_found": entities_found,
                "total_entities": len(entities_found)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PII: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_vault_entry(self, metadata: Dict = None) -> Dict:
        """Create a new vault entry and return vault info"""
        try:
            vault_id = self.vault.create_vault_entry(metadata)
            return {
                "status": "success",
                "vault_id": vault_id,
                "message": "Vault entry created successfully",
                "metadata": metadata or {}
            }
        except Exception as e:
            logger.error(f"Error creating vault entry: {e}")
            return {"status": "error", "message": str(e)}
    
    def anonymize_text(self, text: str, language: str = "en", 
                      entities: Optional[List[str]] = None,
                      anonymization_method: str = "replace",
                      vault_id: str = None,
                      store_in_vault: bool = True) -> Dict:
        """Anonymize PII in text"""
        try:
            # Validate vault_id if provided
            if vault_id and not self.vault.vault_exists(vault_id):
                return {"status": "error", "message": f"Vault ID {vault_id} does not exist"}
            
            # First analyze the text
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=entities
            )
            
            # Configure anonymization operators
            operators = {}
            if anonymization_method == "replace":
                operators = {"DEFAULT": OperatorConfig("replace")}
            elif anonymization_method == "mask":
                operators = {"DEFAULT": OperatorConfig("mask", {"chars_to_mask": 5, "masking_char": "*"})}
            elif anonymization_method == "redact":
                operators = {"DEFAULT": OperatorConfig("redact")}
            elif anonymization_method == "hash":
                operators = {"DEFAULT": OperatorConfig("hash")}
            elif anonymization_method == "encrypt":
                operators = {"DEFAULT": OperatorConfig("encrypt", {"key": "WmZq4t7w!z%C*F-J"})}
            
            # Anonymize the text
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            
            # Create entity mappings for potential de-anonymization
            entity_mappings = {}
            for item in anonymized_result.items:
                original_text = text[item.start:item.end]
                entity_mappings[original_text] = item.text
            
            result = {
                "status": "success",
                "original_text": text,
                "anonymized_text": anonymized_result.text,
                "entities_processed": len(anonymized_result.items),
                "method_used": anonymization_method
            }
            
            # Store in vault if requested
            if store_in_vault:
                used_vault_id = self.vault.store_anonymized_data(
                    original_data=text,
                    anonymized_data=anonymized_result.text,
                    entity_mappings=entity_mappings,
                    metadata={
                        "method": anonymization_method,
                        "language": language,
                        "entities_count": len(analyzer_results)
                    },
                    vault_id=vault_id
                )
                result["vault_id"] = used_vault_id
                result["vault_action"] = "updated" if vault_id else "created"
            
            return result
            
        except Exception as e:
            logger.error(f"Error anonymizing text: {e}")
            return {"status": "error", "message": str(e)}
    
    def deanonymize_from_vault(self, vault_id: str) -> Dict:
        """Retrieve data from vault"""
        try:
            vault_data = self.vault.retrieve_anonymized_data(vault_id)
            if not vault_data:
                return {"status": "error", "message": "Data not found in vault"}
            
            entity_mappings = self.vault.get_entity_mappings(vault_id)
            
            return {
                "status": "success",
                "vault_id": vault_id,
                "anonymized_text": vault_data["anonymized_data"],
                "metadata": vault_data["metadata"],
                "created_at": vault_data["created_at"],
                "entity_mappings": entity_mappings,
                "note": "Full de-anonymization requires additional security measures"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving from vault: {e}")
            return {"status": "error", "message": str(e)}


# Initialize the service
presidio_service = PresidioService()

# FastAPI App
app = FastAPI(
    title="Presidio Anonymization API",
    description="Secure PII anonymization and de-anonymization service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI Routes
@app.get("/")
async def root():
    return {"message": "Presidio Anonymization API", "status": "running"}

@app.post("/vault/create")
async def create_vault_entry(request: CreateVaultRequest):
    """Create a new vault entry and return vault_id"""
    try:
        result = presidio_service.create_vault_entry(request.metadata)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vault/{vault_id}/exists")
async def check_vault_exists(vault_id: str):
    """Check if a vault entry exists"""
    try:
        exists = presidio_service.vault.vault_exists(vault_id)
        return {
            "status": "success",
            "vault_id": vault_id,
            "exists": exists
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "presidio-api"}

@app.post("/analyze")
async def analyze_pii_endpoint(request: AnalyzePIIRequest):
    """Analyze text for PII entities"""
    try:
        result = presidio_service.analyze_pii(
            text=request.text,
            language=request.language,
            entities=request.entities
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anonymize")
async def anonymize_text_endpoint(request: AnonymizeRequest):
    """Anonymize PII in text"""
    try:
        result = presidio_service.anonymize_text(
            text=request.text,
            language=request.language,
            entities=request.entities,
            anonymization_method=request.anonymization_method,
            vault_id=request.vault_id,
            store_in_vault=request.store_in_vault
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deanonymize")
async def deanonymize_endpoint(request: VaultRetrieveRequest):
    """Retrieve and de-anonymize data from vault"""
    try:
        result = presidio_service.deanonymize_from_vault(request.vault_id)
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vault/entries")
async def list_vault_entries(limit: int = 10):
    """List vault entries"""
    try:
        entries = presidio_service.vault.list_entries(limit)
        return {
            "status": "success",
            "entries": entries,
            "total_shown": len(entries)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognizer/custom")
async def configure_custom_recognizer(request: CustomRecognizerRequest):
    """Configure custom recognizer"""
    try:
        return {
            "status": "success",
            "message": "Custom recognizer configuration received",
            "config": request.recognizer_config,
            "note": "Custom recognizer implementation would require additional Presidio setup"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastMCP Server
class PresidioMCPServer:
    """MCP Server for Presidio anonymization services"""
    
    def __init__(self, service: PresidioService):
        self.app = FastMCP("Presidio Anonymization Server")
        self.service = service
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools for anonymization operations"""
        
        @self.app.tool()
        def analyze_pii(text: str, language: str = "en", entities: Optional[List[str]] = None) -> Dict:
            """
            Analyze text for PII entities using Presidio Analyzer
            
            Args:
                text: Text to analyze for PII
                language: Language code (default: en)
                entities: Optional list of specific entities to detect
            
            Returns:
                Dictionary containing detected entities and their details
            """
            return self.service.analyze_pii(text, language, entities)
        
        @self.app.tool()
        def create_vault_entry(metadata: Optional[Dict] = None) -> Dict:
            """
            Create a new vault entry and return vault_id
            
            Args:
                metadata: Optional metadata for the vault entry
            
            Returns:
                Dictionary containing the new vault_id and status
            """
            return self.service.create_vault_entry(metadata)
        
        @self.app.tool()
        def anonymize_text(text: str, language: str = "en", 
                          entities: Optional[List[str]] = None,
                          anonymization_method: str = "replace",
                          vault_id: Optional[str] = None,
                          store_in_vault: bool = True) -> Dict:
            """
            Anonymize PII in text and optionally store in secure vault
            
            Args:
                text: Text to anonymize
                language: Language code (default: en)
                entities: Optional list of specific entities to anonymize
                anonymization_method: Method to use (replace, mask, redact, hash, encrypt)
                vault_id: Optional existing vault_id to store result in
                store_in_vault: Whether to store the result in secure vault
            
            Returns:
                Dictionary containing anonymized text and vault ID if stored
            """
            return self.service.anonymize_text(text, language, entities, anonymization_method, vault_id, store_in_vault)
        
        @self.app.tool()
        def deanonymize_from_vault(vault_id: str) -> Dict:
            """
            Retrieve and attempt to de-anonymize data from vault
            
            Args:
                vault_id: ID of the stored data in vault
            
            Returns:
                Dictionary containing retrieved data and mapping information
            """
            return self.service.deanonymize_from_vault(vault_id)
        
        @self.app.tool()
        def list_vault_entries(limit: int = 10) -> Dict:
            """
            List recent entries in the vault
            
            Args:
                limit: Maximum number of entries to return
            
            Returns:
                Dictionary containing list of vault entries
            """
            try:
                entries = self.service.vault.list_entries(limit)
                return {
                    "status": "success",
                    "entries": entries,
                    "total_shown": len(entries)
                }
            except Exception as e:
                logger.error(f"Error listing vault entries: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.tool()
        def configure_custom_recognizer(recognizer_config: Dict) -> Dict:
            """
            Add a custom PII recognizer to the analyzer
            
            Args:
                recognizer_config: Configuration for custom recognizer including patterns and entities
            
            Returns:
                Dictionary confirming recognizer configuration
            """
            try:
                return {
                    "status": "success",
                    "message": "Custom recognizer configuration received",
                    "config": recognizer_config,
                    "note": "Custom recognizer implementation would require additional Presidio setup"
                }
            except Exception as e:
                logger.error(f"Error configuring custom recognizer: {e}")
                return {"status": "error", "message": str(e)}

    async def run(self, port: int = 8001):
        """Run the MCP server"""
        logger.info(f"Starting Presidio MCP Server on port {port}")
        await self.app.run(port=port)


class DualServer:
    """Manages both FastAPI and FastMCP servers"""
    
    def __init__(self):
        self.fastapi_port = 8000
        self.mcp_port = 8001
        self.mcp_server = PresidioMCPServer(presidio_service)
    
    async def run_fastapi(self):
        """Run FastAPI server"""
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=self.fastapi_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        logger.info(f"Starting FastAPI server on port {self.fastapi_port}")
        await server.serve()
    
    async def run_mcp(self):
        """Run MCP server"""
        await self.mcp_server.run(port=self.mcp_port)
    
    async def run_both(self):
        """Run both servers concurrently"""
        logger.info("Starting Presidio Dual Server (FastAPI + FastMCP)")
        logger.info(f"FastAPI will be available at: http://localhost:{self.fastapi_port}")
        logger.info(f"FastAPI docs will be available at: http://localhost:{self.fastapi_port}/docs")
        logger.info(f"FastMCP will be available at: http://localhost:{self.mcp_port}")
        
        # Run both servers concurrently
        await asyncio.gather(
            self.run_fastapi(),
            self.run_mcp()
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Presidio Dual Server")
    parser.add_argument("--mode", choices=["fastapi", "mcp", "both"], 
                       default="both", help="Server mode to run")
    parser.add_argument("--fastapi-port", type=int, default=8000,
                       help="FastAPI server port")
    parser.add_argument("--mcp-port", type=int, default=8001,
                       help="FastMCP server port")
    
    args = parser.parse_args()
    
    dual_server = DualServer()
    dual_server.fastapi_port = args.fastapi_port
    dual_server.mcp_port = args.mcp_port
    
    try:
        if args.mode == "fastapi":
            asyncio.run(dual_server.run_fastapi())
        elif args.mode == "mcp":
            asyncio.run(dual_server.run_mcp())
        else:  # both
            asyncio.run(dual_server.run_both())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
