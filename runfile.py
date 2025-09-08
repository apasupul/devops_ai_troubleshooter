import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from tachyon_langchain_client import TachyonLangchainClient
from langgraph.prebuilt import create_react_agent

@dataclass
class TicketData:
    """Data structure for ticket information"""
    ticket_id: str
    title: str
    description: str
    created_date: datetime
    priority: str
    assignee: str
    jenkins_links: List[str]
    processed: bool = False

class RCALoopAgent:
    def __init__(self, poll_interval: int = 300):  # 5 minutes default
        self.model_client = TachyonLangchainClient(model_name="gemini-1.5-flash")
        self.agent = None
        self.processed_tickets: Set[str] = set()
        self.poll_interval = poll_interval
        self.running = False
        
        # Server configurations
        self.server_configs = {
            "jira": ["-m", "tachyon_mcp_jira.__main__"],
            "jenkins": ["-m", "tachyon_mcp_jenkins.__main__"],
            "kb_search": ["-m", "tachyon_mcp_knowledge_base.__main__"]  # Assume KB MCP exists
        }
    
    async def initialize_servers(self) -> bool:
        """Initialize all MCP servers and create agent"""
        print("🚀 Initializing RCA Loop Agent...")
        
        all_tools = []
        
        for server_name, args in self.server_configs.items():
            try:
                server_params = StdioServerParameters(command="python", args=args)
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        all_tools.extend(tools)
                        print(f"✅ {server_name}: {len(tools)} tools loaded")
            except Exception as e:
                print(f"❌ Failed to load {server_name} server: {e}")
                return False
        
        if not all_tools:
            print("❌ No tools were loaded. Exiting.")
            return False
        
        # Create agent with all tools
        self.agent = create_react_agent(self.model_client, all_tools)
        print(f"✅ Agent initialized with {len(all_tools)} tools")
        return True
    
    async def poll_jira_for_new_tickets(self) -> List[TicketData]:
        """Poll Jira for new tickets that need RCA"""
        try:
            # Query for tickets created in last poll interval + some buffer
            since_time = datetime.now() - timedelta(seconds=self.poll_interval + 60)
            
            # Use agent to query Jira for new tickets
            jira_query = f"""
            Search for Jira tickets that:
            1. Were created after {since_time.isoformat()}
            2. Have priority High or Critical
            3. Contain keywords: 'error', 'failure', 'exception', 'crash', 'bug'
            4. Are not in 'Closed' or 'Resolved' status
            
            Return the ticket details including ID, title, description, and any Jenkins links.
            """
            
            response = await self.agent.ainvoke({"messages": [jira_query]})
            
            # Parse response to extract ticket data
            tickets = self._parse_jira_response(response)
            
            # Filter out already processed tickets
            new_tickets = [t for t in tickets if t.ticket_id not in self.processed_tickets]
            
            print(f"📋 Found {len(new_tickets)} new tickets to process")
            return new_tickets
            
        except Exception as e:
            print(f"❌ Error polling Jira: {e}")
            return []
    
    def _parse_jira_response(self, response) -> List[TicketData]:
        """Parse agent response to extract ticket data"""
        tickets = []
        try:
            # Extract ticket information from response
            # This would need to be adapted based on actual response format
            content = str(response.get('messages', [{}])[-1].get('content', ''))
            
            # Simple regex patterns to extract ticket info
            ticket_pattern = r'(?:Ticket|Issue)\s+([A-Z]+-\d+)'
            jenkins_pattern = r'https?://[^\s]+jenkins[^\s]*'
            
            ticket_ids = re.findall(ticket_pattern, content, re.IGNORECASE)
            jenkins_links = re.findall(jenkins_pattern, content)
            
            for ticket_id in ticket_ids:
                # For simplicity, creating basic ticket data
                # In reality, you'd parse more detailed info
                ticket = TicketData(
                    ticket_id=ticket_id,
                    title=f"Auto-detected issue in {ticket_id}",
                    description=content[:500] + "..." if len(content) > 500 else content,
                    created_date=datetime.now(),
                    priority="High",
                    assignee="unassigned",
                    jenkins_links=jenkins_links
                )
                tickets.append(ticket)
                
        except Exception as e:
            print(f"❌ Error parsing Jira response: {e}")
        
        return tickets
    
    async def generate_ticket_summary(self, ticket: TicketData) -> Dict[str, str]:
        """Generate summary and search strings for the ticket"""
        summary_prompt = f"""
        Analyze this Jira ticket and provide:
        
        Ticket: {ticket.ticket_id}
        Title: {ticket.title}
        Description: {ticket.description}
        
        Please generate:
        1. A concise summary (2-3 sentences)
        2. Key error keywords for knowledge base search
        3. Potential root cause categories
        4. Search strings for internal documentation
        
        Format as JSON with keys: summary, keywords, categories, search_strings
        """
        
        try:
            response = await self.agent.ainvoke({"messages": [summary_prompt]})
            content = str(response.get('messages', [{}])[-1].get('content', ''))
            
            # Try to parse JSON response
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Fallback parsing
                analysis = {
                    "summary": content[:200] + "...",
                    "keywords": ["error", "failure", "exception"],
                    "categories": ["system error", "application failure"],
                    "search_strings": [ticket.ticket_id, "error logs"]
                }
            
            print(f"📝 Generated analysis for {ticket.ticket_id}")
            return analysis
            
        except Exception as e:
            print(f"❌ Error generating summary for {ticket.ticket_id}: {e}")
            return {}
    
    async def retrieve_jenkins_logs(self, jenkins_links: List[str]) -> Dict[str, str]:
        """Retrieve console logs from Jenkins links"""
        logs = {}
        
        for link in jenkins_links:
            try:
                # Extract build info from link
                build_info = self._extract_build_info(link)
                
                jenkins_query = f"""
                Get the console log for Jenkins build:
                URL: {link}
                Job: {build_info.get('job', 'unknown')}
                Build: {build_info.get('build_number', 'latest')}
                
                Return the full console output, focusing on error messages and stack traces.
                """
                
                response = await self.agent.ainvoke({"messages": [jenkins_query]})
                log_content = str(response.get('messages', [{}])[-1].get('content', ''))
                
                logs[link] = log_content
                print(f"📥 Retrieved logs from {link}")
                
            except Exception as e:
                print(f"❌ Error retrieving logs from {link}: {e}")
                logs[link] = f"Error retrieving logs: {e}"
        
        return logs
    
    def _extract_build_info(self, jenkins_url: str) -> Dict[str, str]:
        """Extract job and build number from Jenkins URL"""
        # Simple regex to extract job and build info
        job_match = re.search(r'/job/([^/]+)', jenkins_url)
        build_match = re.search(r'/(\d+)/?$', jenkins_url)
        
        return {
            'job': job_match.group(1) if job_match else 'unknown',
            'build_number': build_match.group(1) if build_match else 'latest'
        }
    
    async def search_knowledge_base(self, search_strings: List[str], keywords: List[str]) -> str:
        """Search internal knowledge base for relevant information"""
        try:
            # Combine search strings and keywords
            all_terms = search_strings + keywords
            search_query = f"""
            Search the internal knowledge base for information related to:
            Search terms: {', '.join(all_terms)}
            
            Look for:
            - Similar error patterns
            - Resolution procedures
            - Root cause analysis documentation
            - Troubleshooting guides
            
            Return the most relevant documentation and procedures.
            """
            
            response = await self.agent.ainvoke({"messages": [search_query]})
            kb_content = str(response.get('messages', [{}])[-1].get('content', ''))
            
            print(f"🔍 Retrieved knowledge base information")
            return kb_content
            
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return "No relevant documentation found."
    
    async def perform_rca(self, ticket: TicketData, analysis: Dict[str, str], 
                         jenkins_logs: Dict[str, str], kb_info: str) -> str:
        """Perform root cause analysis with all available data"""
        
        rca_prompt = f"""
        Perform a comprehensive Root Cause Analysis for Jira ticket {ticket.ticket_id}.
        
        TICKET INFORMATION:
        - Title: {ticket.title}
        - Description: {ticket.description}
        - Priority: {ticket.priority}
        
        ANALYSIS:
        - Summary: {analysis.get('summary', 'N/A')}
        - Categories: {analysis.get('categories', [])}
        
        JENKINS LOGS:
        {self._format_logs_for_analysis(jenkins_logs)}
        
        KNOWLEDGE BASE INFORMATION:
        {kb_info}
        
        Please provide a detailed RCA including:
        1. **Root Cause**: What caused this issue?
        2. **Impact Assessment**: What systems/users were affected?
        3. **Timeline**: When did this likely start?
        4. **Fix Recommendations**: Immediate and long-term solutions
        5. **Prevention**: How to prevent this in the future
        6. **Confidence Level**: How confident are you in this analysis?
        
        Format as a structured report.
        """
        
        try:
            response = await self.agent.ainvoke({"messages": [rca_prompt]})
            rca_report = str(response.get('messages', [{}])[-1].get('content', ''))
            
            print(f"🔍 Completed RCA for {ticket.ticket_id}")
            return rca_report
            
        except Exception as e:
            print(f"❌ Error performing RCA for {ticket.ticket_id}: {e}")
            return f"Error generating RCA: {e}"
    
    def _format_logs_for_analysis(self, logs: Dict[str, str]) -> str:
        """Format Jenkins logs for RCA analysis"""
        formatted = ""
        for url, log_content in logs.items():
            formatted += f"\n--- JENKINS BUILD: {url} ---\n"
            # Truncate very long logs but keep error sections
            if len(log_content) > 2000:
                error_lines = [line for line in log_content.split('\n') 
                              if any(keyword in line.lower() for keyword in 
                                   ['error', 'exception', 'failed', 'failure', 'stack trace'])]
                formatted += '\n'.join(error_lines[-50:])  # Last 50 error lines
                formatted += f"\n... (truncated from {len(log_content)} characters)\n"
            else:
                formatted += log_content
            formatted += "\n" + "="*50 + "\n"
        
        return formatted
    
    async def save_rca_results(self, ticket: TicketData, rca_report: str):
        """Save RCA results back to Jira or reporting system"""
        try:
            # Update Jira ticket with RCA findings
            update_query = f"""
            Update Jira ticket {ticket.ticket_id} with the following RCA results:
            
            {rca_report}
            
            Also add a comment with timestamp and mark as "RCA Completed".
            Add appropriate labels for tracking.
            """
            
            await self.agent.ainvoke({"messages": [update_query]})
            print(f"💾 Saved RCA results to {ticket.ticket_id}")
            
        except Exception as e:
            print(f"❌ Error saving RCA results for {ticket.ticket_id}: {e}")
    
    async def process_ticket(self, ticket: TicketData):
        """Process a single ticket through the complete RCA workflow"""
        print(f"\n🎫 Processing ticket: {ticket.ticket_id}")
        
        try:
            # Step 1: Generate summary and search terms
            analysis = await self.generate_ticket_summary(ticket)
            
            # Step 2: Retrieve Jenkins logs if available
            jenkins_logs = {}
            if ticket.jenkins_links:
                jenkins_logs = await self.retrieve_jenkins_logs(ticket.jenkins_links)
            
            # Step 3: Search knowledge base
            kb_info = await self.search_knowledge_base(
                analysis.get('search_strings', []),
                analysis.get('keywords', [])
            )
            
            # Step 4: Perform RCA
            rca_report = await self.perform_rca(ticket, analysis, jenkins_logs, kb_info)
            
            # Step 5: Save results
            await self.save_rca_results(ticket, rca_report)
            
            # Mark as processed
            self.processed_tickets.add(ticket.ticket_id)
            
            print(f"✅ Completed RCA for {ticket.ticket_id}")
            
        except Exception as e:
            print(f"❌ Error processing {ticket.ticket_id}: {e}")
    
    async def run_loop(self):
        """Main loop for continuous ticket monitoring"""
        self.running = True
        print(f"🔄 Starting RCA loop (polling every {self.poll_interval} seconds)")
        
        while self.running:
            try:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Polling for new tickets...")
                
                # Poll for new tickets
                new_tickets = await self.poll_jira_for_new_tickets()
                
                # Process each ticket
                for ticket in new_tickets:
                    if not self.running:
                        break
                    await self.process_ticket(ticket)
                
                if not new_tickets:
                    print("😴 No new tickets found")
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Received stop signal")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        print("🏁 RCA loop stopped")
    
    def stop(self):
        """Stop the loop"""
        self.running = False

async def main():
    """Main function to run the RCA loop agent"""
    agent = RCALoopAgent(poll_interval=300)  # Poll every 5 minutes
    
    # Initialize servers
    if not await agent.initialize_servers():
        print("❌ Failed to initialize servers")
        return
    
    try:
        # Start the loop
        await agent.run_loop()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
