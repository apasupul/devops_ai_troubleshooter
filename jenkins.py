#!/usr/bin/env python3
"""
Jenkins MCP Server - Build Instance Agnostic
Retrieves build logs from any Jenkins build URL
"""

import json
import os
import re
import asyncio
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, List
import requests
from requests.auth import HTTPBasicAuth
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JenkinsAuthenticator:
    """Handle different Jenkins authentication methods"""
    
    def __init__(self, config_path: str = "jenkins_config.json"):
        self.config_path = config_path
        self.auth_config = self.load_auth_config()
    
    def load_auth_config(self) -> Dict[str, Any]:
        """Load authentication configuration from file or environment"""
        config = {}
        
        # Try to load from config file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        # Environment variable fallbacks
        if not config:
            config = {
                "default_auth": {
                    "type": "token",
                    "username": os.getenv("JENKINS_USERNAME"),
                    "token": os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_PASSWORD")
                }
            }
        
        return config
    
    def get_auth_for_url(self, jenkins_url: str) -> Optional[requests.auth.AuthBase]:
        """Get appropriate authentication for a Jenkins URL"""
        parsed_url = urlparse(jenkins_url)
        host = parsed_url.netloc.lower()
        
        # Look for specific host configuration
        auth_config = None
        for key, config in self.auth_config.items():
            if key == "default_auth":
                continue
            if host in key.lower() or key.lower() in host:
                auth_config = config
                break
        
        # Fall back to default auth
        if not auth_config:
            auth_config = self.auth_config.get("default_auth", {})
        
        if not auth_config or not auth_config.get("username"):
            logger.warning(f"No authentication configured for {host}")
            return None
        
        auth_type = auth_config.get("type", "basic").lower()
        username = auth_config.get("username")
        
        if auth_type == "token":
            # Jenkins API token authentication
            token = auth_config.get("token")
            if token:
                return HTTPBasicAuth(username, token)
        elif auth_type == "basic":
            # Basic authentication
            password = auth_config.get("password")
            if password:
                return HTTPBasicAuth(username, password)
        
        logger.warning(f"Invalid auth configuration for {host}")
        return None

class JenkinsBuildLogRetriever:
    """Retrieve build logs from Jenkins build URLs"""
    
    def __init__(self):
        self.authenticator = JenkinsAuthenticator()
        self.session = requests.Session()
        # Add common headers
        self.session.headers.update({
            'User-Agent': 'Jenkins-MCP-Server/1.0',
            'Accept': 'application/json, text/plain, */*'
        })
    
    def parse_build_url(self, build_url: str) -> Dict[str, str]:
        """Parse Jenkins build URL to extract components"""
        # Remove trailing slashes and normalize
        build_url = build_url.rstrip('/')
        
        # Handle different Jenkins URL patterns:
        # https://jenkins.example.com/job/my-job/123/
        # https://jenkins.example.com/job/folder/job/my-job/123/
        # https://jenkins.example.com/view/my-view/job/my-job/123/
        
        parsed = urlparse(build_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Extract job path and build number using regex
        path_pattern = r'/(?:view/[^/]+/)?(?:job/([^/]+/?))+([0-9]+)/?'
        match = re.search(path_pattern, parsed.path)
        
        if not match:
            raise ValueError(f"Could not parse Jenkins build URL: {build_url}")
        
        # Extract job name (handle nested folders)
        job_path_parts = parsed.path.split('/job/')
        if len(job_path_parts) < 2:
            raise ValueError(f"Invalid Jenkins job URL structure: {build_url}")
        
        # Get job name (last part before build number)
        job_parts = [part for part in job_path_parts[1:] if part and not part.isdigit()]
        if not job_parts:
            raise ValueError(f"Could not extract job name from URL: {build_url}")
        
        # Build number is the last numeric part
        build_number = match.group(2) if match.group(2) else None
        if not build_number:
            # Try to extract from the end of the path
            path_parts = [p for p in parsed.path.split('/') if p]
            for part in reversed(path_parts):
                if part.isdigit():
                    build_number = part
                    break
        
        if not build_number:
            raise ValueError(f"Could not extract build number from URL: {build_url}")
        
        # Construct job name (join with / for nested folders)
        job_name = '/'.join(job_parts).rstrip('/')
        
        return {
            'base_url': base_url,
            'job_name': job_name,
            'build_number': build_number,
            'full_job_path': '/job/' + '/job/'.join(job_parts).rstrip('/')
        }
    
    def get_build_info(self, build_url: str) -> Dict[str, Any]:
        """Get build information from Jenkins API"""
        parsed = self.parse_build_url(build_url)
        
        # Construct API URL for build info
        api_url = f"{parsed['base_url']}{parsed['full_job_path']}/{parsed['build_number']}/api/json"
        
        auth = self.authenticator.get_auth_for_url(parsed['base_url'])
        
        try:
            response = self.session.get(api_url, auth=auth, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get build info from {api_url}: {e}")
            raise
    
    def get_console_log(self, build_url: str, start_at: int = 0) -> Dict[str, Any]:
        """Get console log from Jenkins build"""
        parsed = self.parse_build_url(build_url)
        
        # Construct console log URL
        console_url = f"{parsed['base_url']}{parsed['full_job_path']}/{parsed['build_number']}/consoleText"
        
        auth = self.authenticator.get_auth_for_url(parsed['base_url'])
        
        try:
            # Add range header if start_at is specified
            headers = {}
            if start_at > 0:
                headers['Range'] = f'bytes={start_at}-'
            
            response = self.session.get(console_url, auth=auth, headers=headers, timeout=60)
            response.raise_for_status()
            
            return {
                'log_text': response.text,
                'size': len(response.text),
                'encoding': response.encoding or 'utf-8',
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get console log from {console_url}: {e}")
            raise
    
    def get_progressive_log(self, build_url: str, start_at: int = 0) -> Dict[str, Any]:
        """Get progressive build log (for ongoing builds)"""
        parsed = self.parse_build_url(build_url)
        
        # Progressive log API endpoint
        log_url = f"{parsed['base_url']}{parsed['full_job_path']}/{parsed['build_number']}/logText/progressiveText"
        
        auth = self.authenticator.get_auth_for_url(parsed['base_url'])
        params = {'start': start_at}
        
        try:
            response = self.session.get(log_url, auth=auth, params=params, timeout=60)
            response.raise_for_status()
            
            # Progressive log includes special headers
            more_data = response.headers.get('X-More-Data', 'false').lower() == 'true'
            text_size = int(response.headers.get('X-Text-Size', 0))
            
            return {
                'log_text': response.text,
                'size': len(response.text),
                'total_size': text_size,
                'has_more': more_data,
                'next_start': text_size,
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get progressive log from {log_url}: {e}")
            raise

class JenkinsMCPServer:
    """Main MCP Server for Jenkins operations"""
    
    def __init__(self):
        self.log_retriever = JenkinsBuildLogRetriever()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available MCP tools"""
        return [
            {
                "name": "get_build_logs",
                "description": "Get console logs from a Jenkins build URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "build_url": {
                            "type": "string",
                            "description": "Full Jenkins build URL (e.g., https://jenkins.example.com/job/my-job/123/)"
                        },
                        "start_at": {
                            "type": "integer",
                            "description": "Start position for log retrieval (default: 0)",
                            "default": 0
                        },
                        "progressive": {
                            "type": "boolean",
                            "description": "Use progressive log API for ongoing builds (default: false)",
                            "default": False
                        }
                    },
                    "required": ["build_url"]
                }
            },
            {
                "name": "get_build_info",
                "description": "Get build information and metadata from a Jenkins build URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "build_url": {
                            "type": "string",
                            "description": "Full Jenkins build URL"
                        }
                    },
                    "required": ["build_url"]
                }
            }
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "get_build_logs":
                return await self.get_build_logs(
                    build_url=arguments["build_url"],
                    start_at=arguments.get("start_at", 0),
                    progressive=arguments.get("progressive", False)
                )
            elif tool_name == "get_build_info":
                return await self.get_build_info(arguments["build_url"])
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {e}")
            return {"error": str(e)}
    
    async def get_build_logs(self, build_url: str, start_at: int = 0, progressive: bool = False) -> Dict[str, Any]:
        """Get build logs (MCP tool implementation)"""
        try:
            if progressive:
                result = self.log_retriever.get_progressive_log(build_url, start_at)
            else:
                result = self.log_retriever.get_console_log(build_url, start_at)
            
            parsed = self.log_retriever.parse_build_url(build_url)
            
            return {
                "success": True,
                "jenkins_instance": parsed["base_url"],
                "job_name": parsed["job_name"],
                "build_number": parsed["build_number"],
                "log_data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_build_info(self, build_url: str) -> Dict[str, Any]:
        """Get build information (MCP tool implementation)"""
        try:
            build_info = self.log_retriever.get_build_info(build_url)
            parsed = self.log_retriever.parse_build_url(build_url)
            
            return {
                "success": True,
                "jenkins_instance": parsed["base_url"],
                "job_name": parsed["job_name"],
                "build_number": parsed["build_number"],
                "build_info": build_info
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Example usage and testing
if __name__ == "__main__":
    # Create example configuration file
    example_config = {
        "jenkins-prod.company.com": {
            "type": "token",
            "username": "api-user",
            "token": "your-api-token-here"
        },
        "jenkins-staging.company.com": {
            "type": "basic",
            "username": "staging-user",
            "password": "staging-password"
        },
        "default_auth": {
            "type": "token",
            "username": "default-user",
            "token": "default-token"
        }
    }
    
    # Save example config
    with open("jenkins_config.json", "w") as f:
        json.dump(example_config, f, indent=2)
    
    print("Example jenkins_config.json created!")
    print("\nTo use the MCP server:")
    print("1. Update jenkins_config.json with your Jenkins credentials")
    print("2. Or set environment variables: JENKINS_USERNAME, JENKINS_TOKEN")
    print("3. Use get_build_logs tool with any Jenkins build URL")
    
    # Example of how to use
    server = JenkinsMCPServer()
    print(f"\nAvailable tools: {[tool['name'] for tool in server.get_available_tools()]}")
