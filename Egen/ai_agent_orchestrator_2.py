import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()

# LLM Client imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class AgentType(Enum):
    STRUCTURAL = "structural"
    PHARMACOKINETIC = "pharmacokinetic"
    CLINICAL = "clinical"

class AnalysisDepth(Enum):
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class VagueLanguageIssue:
    text: str
    category: str
    context: str
    suggestion: str
    severity: str
    confidence: float

@dataclass
class TechnicalIssue:
    issue: str
    category: str
    recommendation: str
    severity: str
    confidence: float
    location: Optional[str] = None

@dataclass
class Conflict:
    statement: str
    resolution: str
    confidence: float
    agents_involved: List[str]

@dataclass
class Recommendation:
    type: str
    recommendation: str
    priority: str
    rationale: str
    confidence: float

@dataclass
class AnalysisResult:
    agent_type: str
    vague_language: List[VagueLanguageIssue]
    technical_issues: List[TechnicalIssue]
    conflicts: List[Conflict]
    recommendations: List[Recommendation]
    confidence_score: float
    processing_time: float
    timestamp: datetime

class LLMClient:
    """Unified LLM client supporting both OpenAI and Anthropic"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)
        elif self.provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)
    
    async def generate_response(self, prompt: str, system_prompt: str = None, max_tokens: int = 4000) -> str:
        """Generate response from LLM"""
        try:
            if self.provider == LLMProvider.OPENAI:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    temperature=0.1,
                    system=system_prompt or "You are a helpful AI assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise Exception(f"LLM generation failed: {str(e)}")

class SpecializedAgent:
    """Base class for specialized AI agents"""
    
    def __init__(self, agent_type: AgentType, llm_client: LLMClient):
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the specific agent type"""
        prompts = {
            AgentType.STRUCTURAL: """
You are a senior structural biologist with expertise in protein structure, drug binding, and molecular interactions. 
Your role is to analyze scientific slide content for structural biology accuracy and clarity.

Focus on:
- Protein structure representations and claims
- Binding site descriptions and druggability assessments
- Molecular interaction mechanisms
- Structural evidence supporting claims
- Homology modeling and structural predictions

Identify vague language, technical inaccuracies, and provide specific recommendations for improvement.
""",
            AgentType.PHARMACOKINETIC: """
You are a pharmacokineticist with deep expertise in ADMET properties, drug metabolism, and toxicology.
Your role is to analyze scientific slide content for pharmacokinetic accuracy and completeness.

Focus on:
- ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties
- Bioavailability and pharmacokinetic parameters
- Drug-drug interactions and metabolic pathways
- Safety profiles and toxicological assessments
- Dose-response relationships

Identify vague language, missing critical information, and provide specific recommendations for improvement.
""",
            AgentType.CLINICAL: """
You are a clinical researcher with expertise in clinical trial design, endpoints, and regulatory requirements.
Your role is to analyze scientific slide content for clinical accuracy and regulatory compliance.

Focus on:
- Clinical trial design and methodology
- Primary and secondary endpoints
- Patient populations and inclusion/exclusion criteria
- Efficacy and safety data presentation
- Regulatory pathway and approval considerations

Identify vague language, unsupported claims, and provide specific recommendations for improvement.
"""
        }
        return prompts.get(self.agent_type, "You are a helpful AI assistant.")
    
    async def analyze_content(self, content: str, slide_type: str, analysis_depth: AnalysisDepth) -> AnalysisResult:
        """Analyze slide content and return structured results"""
        start_time = time.time()
        
        try:
            analysis_prompt = self._create_analysis_prompt(content, slide_type, analysis_depth)
            response = await self.llm_client.generate_response(analysis_prompt, self.system_prompt)
            
            # Parse the structured response
            result = self._parse_analysis_response(response)
            result.processing_time = time.time() - start_time
            result.agent_type = self.agent_type.value
            result.timestamp = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {self.agent_type.value}: {str(e)}")
            # Return error result
            return AnalysisResult(
                agent_type=self.agent_type.value,
                vague_language=[],
                technical_issues=[TechnicalIssue(
                    issue=f"Analysis failed: {str(e)}",
                    category="system_error",
                    recommendation="Please try again or contact support",
                    severity="high",
                    confidence=0.0
                )],
                conflicts=[],
                recommendations=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _create_analysis_prompt(self, content: str, slide_type: str, analysis_depth: AnalysisDepth) -> str:
        """Create analysis prompt based on content and parameters"""
        depth_instructions = {
            AnalysisDepth.STANDARD: "Provide a standard analysis focusing on major issues and clear recommendations.",
            AnalysisDepth.DETAILED: "Provide a detailed analysis with comprehensive issue identification and specific recommendations.",
            AnalysisDepth.COMPREHENSIVE: "Provide a comprehensive analysis with exhaustive issue identification, detailed explanations, and actionable recommendations."
        }
        
        prompt = f"""
Please analyze the following slide content for a {slide_type} presentation.

Analysis Depth: {analysis_depth.value}
Instructions: {depth_instructions[analysis_depth]}

SLIDE CONTENT:
{content}

Please provide your analysis in the following JSON format:
{{
    "vague_language": [
        {{
            "text": "exact vague text found",
            "category": "category of vagueness",
            "context": "surrounding context",
            "suggestion": "specific improvement suggestion",
            "severity": "low|medium|high",
            "confidence": 0.0-1.0
        }}
    ],
    "technical_issues": [
        {{
            "issue": "description of technical issue",
            "category": "category of issue",
            "recommendation": "specific recommendation",
            "severity": "low|medium|high",
            "confidence": 0.0-1.0,
            "location": "where in the slide (optional)"
        }}
    ],
    "conflicts": [
        {{
            "statement": "conflicting statement",
            "resolution": "suggested resolution",
            "confidence": 0.0-1.0,
            "agents_involved": ["current_agent"]
        }}
    ],
    "recommendations": [
        {{
            "type": "improvement|addition|removal|clarification",
            "recommendation": "specific recommendation",
            "priority": "low|medium|high|critical",
            "rationale": "why this recommendation is important",
            "confidence": 0.0-1.0
        }}
    ],
    "confidence_score": 0.0-1.0
}}

Ensure all confidence scores are between 0.0 and 1.0, and provide specific, actionable recommendations.
"""
        return prompt
    
    def _parse_analysis_response(self, response: str) -> AnalysisResult:
        """Parse LLM response into structured AnalysisResult"""
        try:
            # Clean the response to extract JSON
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            # Parse JSON
            data = json.loads(response_clean)
            
            # Convert to dataclass objects
            vague_language = [VagueLanguageIssue(**item) for item in data.get('vague_language', [])]
            technical_issues = [TechnicalIssue(**item) for item in data.get('technical_issues', [])]
            conflicts = [Conflict(**item) for item in data.get('conflicts', [])]
            recommendations = [Recommendation(**item) for item in data.get('recommendations', [])]
            
            return AnalysisResult(
                agent_type=self.agent_type.value,
                vague_language=vague_language,
                technical_issues=technical_issues,
                conflicts=conflicts,
                recommendations=recommendations,
                confidence_score=data.get('confidence_score', 0.5),
                processing_time=0.0,  # Will be set by caller
                timestamp=datetime.now()
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Response content: {response}")
            # Return minimal result on parsing error
            return AnalysisResult(
                agent_type=self.agent_type.value,
                vague_language=[],
                technical_issues=[TechnicalIssue(
                    issue="Failed to parse analysis results",
                    category="parsing_error",
                    recommendation="Please review the analysis manually",
                    severity="medium",
                    confidence=0.0
                )],
                conflicts=[],
                recommendations=[],
                confidence_score=0.0,
                processing_time=0.0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Unexpected parsing error: {str(e)}")
            return AnalysisResult(
                agent_type=self.agent_type.value,
                vague_language=[],
                technical_issues=[TechnicalIssue(
                    issue=f"Analysis parsing failed: {str(e)}",
                    category="system_error",
                    recommendation="Please try again",
                    severity="high",
                    confidence=0.0
                )],
                conflicts=[],
                recommendations=[],
                confidence_score=0.0,
                processing_time=0.0,
                timestamp=datetime.now()
            )

class AIAgentOrchestrator:
    """Main orchestrator class that coordinates multiple AI agents"""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_provider = llm_provider
        self.llm_client = None
        self.agents = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the orchestrator and agents"""
        if self.initialized:
            return
        
        try:
            # Initialize LLM client
            self.llm_client = LLMClient(self.llm_provider)
            
            # Initialize specialized agents
            self.agents = {
                AgentType.STRUCTURAL: SpecializedAgent(AgentType.STRUCTURAL, self.llm_client),
                AgentType.PHARMACOKINETIC: SpecializedAgent(AgentType.PHARMACOKINETIC, self.llm_client),
                AgentType.CLINICAL: SpecializedAgent(AgentType.CLINICAL, self.llm_client)
            }
            
            self.initialized = True
            logger.info(f"AIAgentOrchestrator initialized with {self.llm_provider.value} provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise
    
    async def analyze_slide(self, 
                          slide_content: str, 
                          slide_type: str = "general",
                          agent_types: List[Union[str, AgentType]] = None,
                          analysis_depth: str = "detailed") -> Dict[str, Any]:
        """Main analysis method that coordinates multiple agents"""
        
        if not self.initialized:
            await self.initialize()
        
        # Convert string parameters to enums
        depth_enum = AnalysisDepth(analysis_depth)
        
        # Determine which agents to use
        if agent_types is None or "all" in agent_types:
            selected_agents = list(self.agents.keys())
        else:
            selected_agents = []
            for agent_type in agent_types:
                if isinstance(agent_type, str):
                    try:
                        agent_enum = AgentType(agent_type)
                        selected_agents.append(agent_enum)
                    except ValueError:
                        logger.warning(f"Invalid agent type: {agent_type}")
                elif isinstance(agent_type, AgentType):
                    selected_agents.append(agent_type)
        
        # Run analysis with selected agents
        results = {}
        agent_tasks = []
        
        for agent_type in selected_agents:
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                task = agent.analyze_content(slide_content, slide_type, depth_enum)
                agent_tasks.append((agent_type.value, task))
        
        # Execute all agent analyses concurrently
        for agent_name, task in agent_tasks:
            try:
                result = await task
                results[agent_name] = asdict(result)
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {str(e)}")
                results[agent_name] = {"error": str(e)}
        
        # Perform cross-agent validation
        if len(results) > 1:
            cross_validation = await self._perform_cross_validation(results, slide_content)
            results["cross_validation"] = cross_validation
        
        return results
    
    async def _perform_cross_validation(self, agent_results: Dict[str, Any], slide_content: str) -> Dict[str, Any]:
        """Perform cross-validation between agent results"""
        try:
            # Prepare cross-validation prompt
            validation_prompt = self._create_cross_validation_prompt(agent_results, slide_content)
            
            system_prompt = """
You are a senior scientific reviewer with expertise across structural biology, pharmacokinetics, and clinical research.
Your role is to cross-validate analyses from multiple specialized agents and identify consensus, conflicts, and priority recommendations.
"""
            
            response = await self.llm_client.generate_response(validation_prompt, system_prompt)
            
            # Parse cross-validation response
            return self._parse_cross_validation_response(response)
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return {
                "consistency_score": 0.0,
                "consensus_issues": [],
                "conflicting_findings": [],
                "priority_recommendations": [],
                "error": str(e)
            }
    
    def _create_cross_validation_prompt(self, agent_results: Dict[str, Any], slide_content: str) -> str:
        """Create prompt for cross-validation analysis"""
        results_summary = {}
        for agent_name, result in agent_results.items():
            if "error" not in result:
                results_summary[agent_name] = {
                    "issues_count": len(result.get("technical_issues", [])),
                    "recommendations_count": len(result.get("recommendations", [])),
                    "confidence": result.get("confidence_score", 0.0),
                    "key_issues": [issue.get("issue", "") for issue in result.get("technical_issues", [])[:3]]
                }
        
        prompt = f"""
Please perform cross-validation analysis on the following agent results for a scientific slide.

SLIDE CONTENT:
{slide_content}

AGENT RESULTS SUMMARY:
{json.dumps(results_summary, indent=2)}

Please provide your cross-validation analysis in the following JSON format:
{{
    "consistency_score": 0.0-1.0,
    "consensus_issues": [
        {{
            "category": "issue category",
            "description": "consensus description",
            "agent_count": number_of_agents_that_found_this,
            "severity": "low|medium|high"
        }}
    ],
    "conflicting_findings": [
        {{
            "type": "conflict type",
            "conflict": "description of conflict",
            "agents_involved": ["agent1", "agent2"],
            "resolution": "suggested resolution"
        }}
    ],
    "priority_recommendations": [
        {{
            "priority": "critical|high|medium|low",
            "recommendation": "consolidated recommendation",
            "rationale": "why this is priority",
            "supporting_agents": ["agent1", "agent2"]
        }}
    ]
}}
"""
        return prompt
    
    def _parse_cross_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse cross-validation response"""
        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            data = json.loads(response_clean)
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Cross-validation JSON parsing error: {str(e)}")
            return {
                "consistency_score": 0.0,
                "consensus_issues": [],
                "conflicting_findings": [],
                "priority_recommendations": [],
                "parsing_error": str(e)
            }
        except Exception as e:
            logger.error(f"Cross-validation parsing error: {str(e)}")
            return {
                "consistency_score": 0.0,
                "consensus_issues": [],
                "conflicting_findings": [],
                "priority_recommendations": [],
                "error": str(e)
            }

# Example usage and testing
async def main():
    """Example usage of the AIAgentOrchestrator"""
    
    # Example slide content
    sample_content = """
    Our novel compound XYZ-123 shows promising binding affinity to the target protein.
    The compound exhibits good bioavailability and minimal side effects.
    Clinical trials demonstrate significant efficacy in patient populations.
    The mechanism involves selective inhibition of the target enzyme.
    """
    
    try:
        # Initialize orchestrator (requires API key in environment)
        orchestrator = AIAgentOrchestrator(llm_provider=LLMProvider.OPENAI)
        await orchestrator.initialize()
        
        # Run analysis
        results = await orchestrator.analyze_slide(
            slide_content=sample_content,
            slide_type="compound_profile",
            agent_types=["all"],
            analysis_depth="detailed"
        )
        
        # Print results
        print("Analysis Results:")
        print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())