#!/usr/bin/env python3

import json
import httpx
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Jenkins Log Extractor")


@mcp.tool()
def extract_jenkins_logs(
    url: str,
    type: str = "EFL",
    max_lines: int = 100,
    action: str = "log-extractor"
) -> str:
    """
    Extract console logs from Jenkins builds using the EFL/Jenkins pipeline log extractor API.
    
    Args:
        url: Jenkins build URL (e.g., https://jenkins-build12.wellsfargo.net/job/utilities-ejen/job/Utilities-ejen-epl-pipeline-log-extractor/job/poc/11/)
        type: Log extraction type (default: EFL)
        max_lines: Maximum number of lines to extract from console (default: 100)
        action: Action to perform (default: log-extractor)
    
    Returns:
        Formatted results from the Jenkins log extraction API
    """
    
    # API endpoint
    api_endpoint = "https://ejen-epl-pipeline-log-extractor-dev.apps.gar09.ocp.nonprod.wellsfargo.net/v1/epl/pipelinelogextractor"
    
    # Prepare payload
    payload = {
        "url": url,
        "type": type,
        "max_lines": max_lines,
        "action": action
    }
    
    try:
        # Make the API request
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
        
        # Format the response
        response_text = f"""Jenkins Log Extraction Results:

URL: {url}
Type: {type}
Max Lines: {max_lines}
Action: {action}

Response:
{json.dumps(result, indent=2)}"""
        
        return response_text
        
    except httpx.HTTPStatusError as e:
        return f"HTTP Error {e.response.status_code}: {e.response.text}"
    except httpx.RequestError as e:
        return f"Request failed: {str(e)}"
    except Exception as e:
        return f"Failed to extract Jenkins logs: {str(e)}"


if __name__ == "__main__":
    mcp.run()
