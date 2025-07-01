import time
from typing import List, Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.qa.llamaindex_supervisor import LlamaIndexSupervisorAgent
from database.models import AgentResponse, UserQuery, QueryIntent, QueryResponse
from utils.logger import logger

class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Supervisor",
            description="Orchestrates query processing using LlamaIndex ReAct agent framework"
        )
        
        # Use LlamaIndex supervisor for advanced agentic capabilities
        self.llamaindex_supervisor = LlamaIndexSupervisorAgent()
    
    async def process(self, query: UserQuery, **kwargs) -> AgentResponse:
        """Process query using LlamaIndex supervisor agent."""
        try:
            self.log_activity(f"Delegating to LlamaIndex supervisor: {query.query_text}")
            
            # Delegate to LlamaIndex supervisor
            response = await self.llamaindex_supervisor.process(query, **kwargs)
            
            # Update the agent name to maintain compatibility
            response.agent_name = self.name
            
            return response
            
        except Exception as e:
            self.log_activity(f"LlamaIndex supervisor failed: {e}", "error")
            processing_time = time.time() - time.time()
            
            return self.create_response(
                content=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
