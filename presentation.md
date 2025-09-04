<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Architecture for Chat Application</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            overflow-x: hidden;
        }

        .presentation-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .slide {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin: 20px 0;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .slide:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        }

        .slide-title {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .slide-subtitle {
            font-size: 1.8em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            color: #4a5568;
        }

        .slide-content {
            font-size: 1.1em;
            line-height: 1.6;
            color: #4a5568;
        }

        .architecture-diagram {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            border: 2px solid #e2e8f0;
            position: relative;
            overflow: hidden;
        }

        .component {
            background: linear-gradient(135deg, #4299e1, #3182ce);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 10px;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
            transition: transform 0.3s ease;
        }

        .component:hover {
            transform: scale(1.05);
        }

        .component.mcp-host {
            background: linear-gradient(135deg, #ed8936, #dd6b20);
            box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
        }

        .component.agent-server {
            background: linear-gradient(135deg, #38a169, #2f855a);
            box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3);
        }

        .component.service {
            background: linear-gradient(135deg, #805ad5, #6b46c1);
            box-shadow: 0 4px 15px rgba(128, 90, 213, 0.3);
        }

        .flow-diagram {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }

        .flow-arrow {
            text-align: center;
            font-size: 2em;
            color: #667eea;
            margin: 10px 0;
        }

        .code-block {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }

        .highlight {
            background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }

        .benefits-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .benefit-card {
            background: linear-gradient(135deg, #fff5f5, #fed7d7);
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid #e53e3e;
        }

        .tech-stack {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .tech-item {
            background: linear-gradient(135deg, #f0fff4, #c6f6d5);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #38a169;
        }

        .implementation-steps {
            counter-reset: step-counter;
        }

        .step {
            counter-increment: step-counter;
            background: linear-gradient(135deg, #ebf8ff, #bee3f8);
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            border-left: 4px solid #3182ce;
            position: relative;
        }

        .step::before {
            content: counter(step-counter);
            position: absolute;
            left: -15px;
            top: 50%;
            transform: translateY(-50%);
            background: #3182ce;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        @media (max-width: 768px) {
            .slide {
                padding: 20px;
                margin: 10px 0;
            }
            
            .slide-title {
                font-size: 2em;
            }
            
            .flow-diagram {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="presentation-container">
        <!-- Title Slide -->
        <div class="slide">
            <h1 class="slide-title">Building an MCP Architecture</h1>
            <h2 style="font-size: 1.5em; color: #667eea; text-align: center; margin-bottom: 30px;">
                For Chat Application with LLM Integration
            </h2>
            <div style="text-align: center; font-size: 1.2em; color: #4a5568;">
                <p><strong>Connecting to Jira • Jenkins • External Services</strong></p>
                <p style="margin-top: 20px; font-style: italic;">A comprehensive guide to Model Context Protocol implementation</p>
            </div>
        </div>

        <!-- What is MCP Slide -->
        <div class="slide">
            <h2 class="slide-title">What is MCP?</h2>
            <div class="slide-content">
                <div class="highlight">
                    <strong>Model Context Protocol (MCP)</strong> is an open standard for connecting AI assistants to data sources and tools, enabling secure and controlled access to external resources.
                </div>
                
                <h3 class="slide-subtitle">Key Concepts</h3>
                <p><strong>Resources:</strong> Data that models can read (files, API responses, database records)</p>
                <p><strong>Tools:</strong> Functions that models can execute (API calls, system commands)</p>
                <p><strong>Prompts:</strong> Reusable prompt templates with arguments</p>
                
                <h3 class="slide-subtitle">Why MCP for Chat Applications?</h3>
                <p>• <strong>Standardized Integration:</strong> Consistent way to connect to multiple services</p>
                <p>• <strong>Security:</strong> Controlled access to external resources</p>
                <p>• <strong>Scalability:</strong> Easy to add new integrations</p>
                <p>• <strong>Maintainability:</strong> Decoupled architecture</p>
            </div>
        </div>

        <!-- Architecture Overview Slide -->
        <div class="slide">
            <h2 class="slide-title">Architecture Overview</h2>
            <div class="architecture-diagram">
                <div class="flow-diagram">
                    <div>
                        <div class="component">Chat Frontend</div>
                        <div class="flow-arrow">↓</div>
                        <div class="component mcp-host">MCP Host</div>
                    </div>
                    <div>
                        <div class="component">LLM Service</div>
                        <div class="flow-arrow">↔</div>
                        <div class="component agent-server">Agent Server</div>
                    </div>
                    <div>
                        <div class="component service">MCP Servers</div>
                        <div class="flow-arrow">↓</div>
                        <div class="component service">Jira • Jenkins</div>
                    </div>
                </div>
            </div>
            
            <div class="slide-content">
                <h3 class="slide-subtitle">Data Flow</h3>
                <p>1. User sends message through chat interface</p>
                <p>2. MCP Host processes and routes to appropriate services</p>
                <p>3. Agent Server coordinates with LLM and MCP servers</p>
                <p>4. MCP Servers interact with external services (Jira, Jenkins)</p>
                <p>5. Response flows back through the chain to user</p>
            </div>
        </div>

        <!-- Core Components Slide -->
        <div class="slide">
            <h2 class="slide-title">Core Components Explained</h2>
            
            <h3 class="slide-subtitle">MCP Host</h3>
            <div class="highlight">
                <strong>Purpose:</strong> Central coordinator that manages MCP client connections and server discovery
                <br><strong>Responsibilities:</strong>
                <p>• Server lifecycle management (start, stop, restart MCP servers)</p>
                <p>• Connection pooling and load balancing</p>
                <p>• Authentication and authorization</p>
                <p>• Request routing and response aggregation</p>
            </div>

            <h3 class="slide-subtitle">Agent Server</h3>
            <div class="highlight">
                <strong>Purpose:</strong> Orchestrates interactions between LLM and various MCP servers
                <br><strong>Responsibilities:</strong>
                <p>• LLM conversation management</p>
                <p>• Tool selection and execution planning</p>
                <p>• Context management and memory</p>
                <p>• Response formatting and validation</p>
            </div>

            <h3 class="slide-subtitle">MCP Servers</h3>
            <div class="highlight">
                <strong>Purpose:</strong> Service-specific implementations that expose resources and tools
                <br><strong>Examples:</strong>
                <p>• <strong>Jira MCP Server:</strong> Issue management, project data, workflows</p>
                <p>• <strong>Jenkins MCP Server:</strong> Build management, pipeline control, job status</p>
                <p>• <strong>Database MCP Server:</strong> Query execution, data retrieval</p>
            </div>
        </div>

        <!-- Jira Integration Slide -->
        <div class="slide">
            <h2 class="slide-title">Jira MCP Server</h2>
            
            <h3 class="slide-subtitle">Resources Exposed</h3>
            <div class="tech-stack">
                <div class="tech-item">
                    <strong>Projects</strong><br>
                    Project metadata, configurations
                </div>
                <div class="tech-item">
                    <strong>Issues</strong><br>
                    Tickets, bugs, tasks, stories
                </div>
                <div class="tech-item">
                    <strong>Workflows</strong><br>
                    Status transitions, approvals
                </div>
                <div class="tech-item">
                    <strong>Users & Permissions</strong><br>
                    Team members, roles, access
                </div>
            </div>

            <h3 class="slide-subtitle">Tools Available</h3>
            <div class="code-block">
create_issue(project, summary, description, issue_type, assignee)
update_issue(issue_key, fields)
transition_issue(issue_key, transition_id)
search_issues(jql_query, max_results)
add_comment(issue_key, comment)
get_project_info(project_key)
list_workflows(project_key)
            </div>

            <h3 class="slide-subtitle">Example Use Cases</h3>
            <p>• "Create a bug report for the login issue"</p>
            <p>• "Show me all high-priority tickets assigned to me"</p>
            <p>• "Move ticket ABC-123 to 'In Progress'"</p>
            <p>• "What's the status of the user authentication project?"</p>
        </div>

        <!-- Jenkins Integration Slide -->
        <div class="slide">
            <h2 class="slide-title">Jenkins MCP Server</h2>
            
            <h3 class="slide-subtitle">Resources Exposed</h3>
            <div class="tech-stack">
                <div class="tech-item">
                    <strong>Jobs</strong><br>
                    Build configurations, pipelines
                </div>
                <div class="tech-item">
                    <strong>Builds</strong><br>
                    Build history, logs, artifacts
                </div>
                <div class="tech-item">
                    <strong>Nodes</strong><br>
                    Agent status, capacity
                </div>
                <div class="tech-item">
                    <strong>Plugins</strong><br>
                    Installed plugins, configurations
                </div>
            </div>

            <h3 class="slide-subtitle">Tools Available</h3>
            <div class="code-block">
trigger_build(job_name, parameters)
get_build_status(job_name, build_number)
get_build_logs(job_name, build_number)
stop_build(job_name, build_number)
list_jobs(folder_path)
get_job_config(job_name)
create_job(job_name, config_xml)
disable_job(job_name)
            </div>

            <h3 class="slide-subtitle">Example Use Cases</h3>
            <p>• "Trigger a deployment build for the staging environment"</p>
            <p>• "What's the status of the latest main branch build?"</p>
            <p>• "Show me the test results from build #145"</p>
            <p>• "Cancel the running build for the feature branch"</p>
        </div>

        <!-- Technical Implementation Slide -->
        <div class="slide">
            <h2 class="slide-title">Technical Implementation</h2>
            
            <h3 class="slide-subtitle">Technology Stack</h3>
            <div class="tech-stack">
                <div class="tech-item">
                    <strong>MCP Host</strong><br>
                    Node.js + Express<br>
                    WebSocket support
                </div>
                <div class="tech-item">
                    <strong>Agent Server</strong><br>
                    Python + FastAPI<br>
                    LangChain/LlamaIndex
                </div>
                <div class="tech-item">
                    <strong>MCP Servers</strong><br>
                    Python MCP SDK<br>
                    Service-specific APIs
                </div>
                <div class="tech-item">
                    <strong>Chat Frontend</strong><br>
                    React + WebSocket<br>
                    Real-time updates
                </div>
            </div>

            <h3 class="slide-subtitle">MCP Server Example (Jira)</h3>
            <div class="code-block">
from mcp.server import Server
from mcp.types import Resource, Tool
import jira

server = Server("jira-mcp")

@server.resource("jira://issues/{project_key}")
async def get_project_issues(project_key: str):
    # Return project issues as MCP resource
    pass

@server.tool("create_issue")
async def create_issue(project: str, summary: str, 
                      description: str, issue_type: str):
    # Create Jira issue and return result
    pass

if __name__ == "__main__":
    server.run()
            </div>
        </div>

        <!-- Implementation Steps Slide -->
        <div class="slide">
            <h2 class="slide-title">Implementation Roadmap</h2>
            
            <div class="implementation-steps">
                <div class="step">
                    <strong>Phase 1: Core Infrastructure</strong><br>
                    Set up MCP Host, basic Agent Server, and chat frontend with WebSocket communication
                </div>
                
                <div class="step">
                    <strong>Phase 2: LLM Integration</strong><br>
                    Integrate chosen LLM (OpenAI, Anthropic, local model) with conversation management
                </div>
                
                <div class="step">
                    <strong>Phase 3: First MCP Server</strong><br>
                    Implement Jira MCP server with basic issue management capabilities
                </div>
                
                <div class="step">
                    <strong>Phase 4: Jenkins Integration</strong><br>
                    Add Jenkins MCP server for build management and CI/CD operations
                </div>
                
                <div class="step">
                    <strong>Phase 5: Advanced Features</strong><br>
                    Add authentication, logging, monitoring, and error handling
                </div>
                
                <div class="step">
                    <strong>Phase 6: Testing & Optimization</strong><br>
                    Comprehensive testing, performance optimization, and security hardening
                </div>
            </div>
        </div>

        <!-- Benefits Slide -->
        <div class="slide">
            <h2 class="slide-title">Benefits of MCP Architecture</h2>
            
            <div class="benefits-grid">
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">🔧 Modularity</h4>
                    <p>Each service integration is independent, making development and maintenance easier</p>
                </div>
                
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">🔒 Security</h4>
                    <p>Controlled access with authentication and authorization at multiple layers</p>
                </div>
                
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">📈 Scalability</h4>
                    <p>Easy to add new service integrations without affecting existing ones</p>
                </div>
                
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">🔄 Standardization</h4>
                    <p>Consistent interface for all external service interactions</p>
                </div>
                
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">🐛 Error Handling</h4>
                    <p>Isolated failure domains - one service failure doesn't break others</p>
                </div>
                
                <div class="benefit-card">
                    <h4 style="color: #e53e3e; margin-bottom: 10px;">⚡ Performance</h4>
                    <p>Parallel processing and caching capabilities for multiple service calls</p>
                </div>
            </div>
        </div>

        <!-- Next Steps Slide -->
        <div class="slide">
            <h2 class="slide-title">Next Steps</h2>
            
            <div class="slide-content">
                <h3 class="slide-subtitle">Immediate Actions</h3>
                <p>1. <strong>Environment Setup:</strong> Install MCP SDK and development tools</p>
                <p>2. <strong>Service Accounts:</strong> Create API credentials for Jira and Jenkins</p>
                <p>3. <strong>Prototype:</strong> Build minimal viable product with one integration</p>
                
                <h3 class="slide-subtitle">Architecture Decisions</h3>
                <p>• <strong>LLM Choice:</strong> OpenAI GPT-4, Claude, or local model?</p>
                <p>• <strong>Hosting:</strong> Cloud (AWS, GCP, Azure) or on-premises?</p>
                <p>• <strong>Database:</strong> PostgreSQL, MongoDB, or Redis for caching?</p>
                <p>• <strong>Monitoring:</strong> Prometheus/Grafana, ELK stack, or cloud solutions?</p>
                
                <div class="highlight" style="margin-top: 30px;">
                    <strong>Success Metrics:</strong>
                    <p>• Response time &lt; 3 seconds for simple queries</p>
                    <p>• 99.9% uptime for core services</p>
                    <p>• Support for 10+ concurrent users initially</p>
                    <p>• Extensible to 5+ service integrations</p>
                </div>
            </div>
        </div>

        <!-- Q&A Slide -->
        <div class="slide">
            <h2 class="slide-title">Questions & Discussion</h2>
            <div style="text-align: center; margin-top: 50px;">
                <div style="font-size: 4em; margin-bottom: 30px;">🤔</div>
                <h3 style="color: #667eea; margin-bottom: 20px;">Ready to build your MCP-powered chat application?</h3>
                <p style="font-size: 1.2em; color: #4a5568;">Let's discuss implementation details, technical challenges, and next steps!</p>
            </div>
        </div>
    </div>
</body>
</html>