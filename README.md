🤖 Multi-Agent AI System for Business Intelligence
🌟 Features
Core Capabilities

🤖 5 Specialized AI Agents - Each with unique personality and expertise

Supervisor Agent - Orchestrates multi-agent workflows
Data Analyst - SQL queries, statistical analysis, trend detection
Researcher - Information gathering, competitive analysis, web research
Strategist - Strategic planning, scenario analysis, recommendations
Validator - Quality control, fact-checking, error detection


💬 Advanced Communication System

Priority-based message queuing (Redis)
Fire-and-forget & request-response patterns
Real-time pub/sub broadcasts
Full message tracking and audit trails


🧠 Three-Tier Memory Architecture

Short-term - Redis working memory (seconds to hours)
Long-term - PostgreSQL persistent storage (days to forever)
Episodic - Experience-based learning from past successes/failures


🛠️ 20+ Specialized Tools

SQL execution with query optimization
Statistical analysis (correlation, regression, hypothesis testing)
Data visualization generation
Web research simulation
File I/O operations


🔍 Semantic Search

Vector embeddings with pgvector
Similarity-based memory retrieval
RAG (Retrieval-Augmented Generation)


📊 Real-time Dashboard

React + TypeScript frontend
Live agent monitoring
Task queue visualization
Performance metrics and analytics




🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                         USER REQUEST                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                          │
│  • Task decomposition                                        │
│  • Agent orchestration                                       │
│  • Response synthesis                                        │
└───┬──────────────┬──────────────┬──────────────┬────────────┘
    ↓              ↓              ↓              ↓
┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐
│  DATA   │  │RESEARCHER│  │STRATEGIST │  │VALIDATOR │
│ ANALYST │  │  AGENT   │  │  AGENT    │  │  AGENT   │
└────┬────┘  └─────┬────┘  └─────┬─────┘  └─────┬────┘
     │             │              │              │
     └─────────────┴──────────────┴──────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   SHARED INFRASTRUCTURE                      │
│  • Message Queue (Redis)                                     │
│  • Memory Systems (Redis + PostgreSQL + pgvector)           │
│  • Tool Execution Engine                                     │
│  • LLM Backend (Ollama)                                      │
└─────────────────────────────────────────────────────────────┘
Technology Stack
LayerTechnologyPurposeLLMOllama (Llama 3.2, Mistral)Local AI inferenceBackendPython 3.11, FastAPIAPI & business logicFrontendReact 18, TypeScript, TailwindCSSUser interfaceMessage QueueRedis 7Inter-agent communicationDatabasePostgreSQL 15 + pgvectorPersistent storage & vectorsBuild ToolViteFrontend bundlingOrchestrationLangChainAgent frameworkDeploymentDocker ComposeContainerization

🚀 Quick Start
Prerequisites

Python 3.11+ - Download
Node.js 18+ - Download
Docker Desktop - Download
Ollama - Download
Git - Download

📊 Project Structure
multi-agent-business-intelligence/
│
├── src/                       # Backend (Python)
│   ├── agents/                # Agent implementations
│   │   ├── base_agent.py      # Abstract base class
│   │   ├── data_analyst_agent.py
│   │   ├── researcher_agent.py
│   │   ├── strategist_agent.py
│   │   ├── validator_agent.py
│   │   └── supervisor_agent.py
│   │
│   ├── communication/         # Message passing system
│   │   ├── protocol.py        # Redis queuing
│   │   └── conversation_memory.py
│   │
│   ├── memory/                # Memory systems
│   │   ├── short_term_memory.py    # Redis
│   │   ├── long_term_memory.py     # PostgreSQL
│   │   ├── episodic_memory.py
│   │   └── vector_store.py         # Semantic search
│   │
│   ├── tools/                 # Agent capabilities
│   ├── llm/                   # Ollama integration
│   ├── orchestration/         # Workflow engine
│   ├── api/                   # FastAPI endpoints
│   ├── config/                # Configuration
│   └── utils/                 # Utilities
│
├── frontend/                  # Frontend (TypeScript/React)
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── hooks/             # Custom hooks
│   │   ├── services/          # API clients
│   │   ├── types/             # TypeScript types
│   │   └── utils/             # Utilities
│   │
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/                   # Utility scripts
│   ├── setup_database.py
│   ├── test_installation.py
│   └── seed_data.py
│
├── docs/                      # Documentation
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md

📚 Documentation
Available Documentation

Architecture Guide - System design and patterns
API Reference - Complete API documentation
Agent Development Guide - Creating custom agents
Deployment Guide - Production deployment
Troubleshooting - Common issues and solutions

Key Concepts
Agents
Each agent has:

Role - What type of agent (analyst, researcher, etc.)
Personality - Behavior traits (risk tolerance, creativity, etc.)
Capabilities - Skills/tools it can use
System Prompt - Instructions for LLM behavior



📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

LangChain - Agent framework foundation
Ollama - Local LLM inference
FastAPI - High-performance API framework
React - Frontend framework
pgvector - Vector similarity search
Redis - Message queuing and caching
PostgreSQL - Reliable data storage
