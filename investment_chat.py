"""
Investment Strategy Chat – Warren Buffett & George Soros
A chat interface where you can discuss investment strategies with AI versions of legendary investors

Run with:
    pip install agno openai
    python investment_chat.py

Environment:
  • Set your OpenAI key in .env or as environment variable →  OPENAI_API_KEY="sk‑..."

Features:
  • Chat with Warren Buffett about value investing, economic moats, and long-term strategies
  • Chat with George Soros about reflexivity theory, macro trading, and market psychology
  • Switch between advisors or ask both simultaneously
  • Get personalized investment advice and philosophical insights
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Agno imports
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

# Terminal colors for better UI
try:
    from colorama import init, Fore, Back, Style
    init()
    COLOR_ENABLED = True
except ImportError:
    # Fallback if colorama is not installed
    class DummyFore:
        def __getattr__(self, name):
            return ""
    class DummyBack:
        def __getattr__(self, name):
            return ""
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    Fore = DummyFore()
    Back = DummyBack()
    Style = DummyStyle()
    COLOR_ENABLED = False

# Load environment variables
load_dotenv()

# --- PYDANTIC MODELS ---------------------------------------------------------
class InvestmentAdvice(BaseModel):
    response: str = Field(..., description="The investor's response to the question or comment")
    key_principles: list[str] = Field(..., description="Key investment principles mentioned in the response")
    recommended_action: str = Field(..., description="Specific recommended action or next step, if any")

# --- AGNO AGENT SETUP --------------------------------------------------------
OPENAI_API_KEY_VALUE = None
AGNO_READY = False
buffett_agent = None
soros_agent = None

def initialize_agno_agents():
    """Initialize Agno agents for Warren Buffett and George Soros."""
    global OPENAI_API_KEY_VALUE, AGNO_READY, buffett_agent, soros_agent
    
    try:
        import agno
    except ImportError:
        print("ERROR: 'agno' package not installed. Install with: pip install agno")
        return False
    
    OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY_VALUE:
        print("ERROR: OpenAI API key not found. Set it in .env file or as environment variable.")
        return False
    
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_VALUE
    
    try:
        # Warren Buffett Agent - focused on value investing and long-term wisdom
        buffett_agent = Agent(
            name="WarrenBuffettChat",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are Warren Buffett, the legendary value investor known as the 'Oracle of Omaha'.",
                "Respond in your characteristic folksy, wise, and humble style with occasional humor and metaphors.",
                "Share insights about value investing, economic moats, business fundamentals, and long-term thinking.",
                "Use your famous quotes and analogies when appropriate (baseball, buying businesses, etc.).",
                "Emphasize patience, discipline, and the importance of understanding what you're investing in.",
                "Discuss concepts like intrinsic value, margin of safety, competitive advantages, and quality management.",
                "Be encouraging but realistic about the challenges of investing.",
                "Always consider the questioner as someone you're mentoring - be generous with wisdom but not preachy.",
                "Reference your experiences with Berkshire Hathaway, Charlie Munger, and notable investments when relevant.",
                "Keep responses conversational and accessible, avoiding overly technical jargon."
            ],
            response_model=InvestmentAdvice,
        )
        
        # George Soros Agent - focused on macro trading and reflexivity theory
        soros_agent = Agent(
            name="GeorgeSorosChat",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are George Soros, the legendary macro trader and philosopher of markets.",
                "Respond with intellectual depth, discussing market psychology, reflexivity theory, and global macro trends.",
                "Share insights about market inefficiencies, boom-bust cycles, and the interplay between perception and reality.",
                "Explain reflexivity - how market participants' perceptions influence reality, which in turn affects perceptions.",
                "Discuss themes like financial bubbles, currency speculation, central bank policies, and political economy.",
                "Be philosophical and analytical, often connecting markets to broader social and political trends.",
                "Reference concepts like fallibility of human understanding, uncertainty, and the limits of economic theory.",
                "Mention experiences with quantum fund, currency trades (like breaking the Bank of England), and market crises.",
                "Be intellectually rigorous while remaining conversational and thought-provoking.",
                "Emphasize the importance of adaptability and recognizing when you're wrong."
            ],
            response_model=InvestmentAdvice,
        )
        
        AGNO_READY = True
        return True
        
    except Exception as e:
        print(f"Failed to initialize Agno agents: {e}")
        return False

# --- CLI INTERFACE -----------------------------------------------------------
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header."""
    print("\n" + Fore.CYAN + "=" * 80 + Style.RESET_ALL)
    print(Fore.CYAN + " " * 20 + Style.BRIGHT + "INVESTMENT STRATEGY CHAT" + Style.RESET_ALL)
    print(Fore.CYAN + " " * 15 + "Chat with Warren Buffett & George Soros" + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 80 + Style.RESET_ALL)

def print_menu():
    """Print the main menu options."""
    print(f"\n{Fore.YELLOW + Style.BRIGHT}Choose your advisor:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}1. Warren Buffett{Style.RESET_ALL} - Value investing & long-term strategies")
    print(f"{Fore.MAGENTA}2. George Soros{Style.RESET_ALL} - Macro trading & reflexivity theory")
    print(f"{Fore.CYAN}3. Ask Both{Style.RESET_ALL} - Get perspectives from both legends")
    print(f"{Fore.WHITE}4. Investment Topics{Style.RESET_ALL} - Suggested conversation starters")
    print(f"{Fore.RED}5. Exit{Style.RESET_ALL}")

def print_topics():
    """Print suggested investment topics for discussion."""
    print(f"\n{Fore.YELLOW + Style.BRIGHT}Suggested Investment Topics:{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}Warren Buffett Topics:{Style.RESET_ALL}")
    print("• How to identify companies with economic moats")
    print("• The importance of understanding business fundamentals")
    print("• Value investing in today's market environment")
    print("• Building a long-term investment philosophy")
    print("• Evaluating management quality and corporate culture")
    print("• The role of patience and discipline in investing")
    
    print(f"\n{Fore.MAGENTA}George Soros Topics:{Style.RESET_ALL}")
    print("• Understanding market reflexivity and feedback loops")
    print("• Identifying and trading financial bubbles")
    print("• The impact of central bank policies on markets")
    print("• Currency trading and global macro strategies")
    print("• Market psychology and behavioral finance")
    print("• Adapting to changing market conditions")
    
    print(f"\n{Fore.CYAN}General Investment Topics:{Style.RESET_ALL}")
    print("• Current market opportunities and risks")
    print("• Portfolio construction and risk management")
    print("• The future of investing and market evolution")
    print("• Lessons from major market crises")
    print("• Technology's impact on traditional investing")
    print("• ESG investing and sustainable finance")

def chat_with_buffett(question):
    """Get Warren Buffett's response to a question."""
    if not AGNO_READY or not buffett_agent:
        return {
            "response": "I apologize, but I'm not available for chat right now due to technical difficulties. Please check your OpenAI API key setup.",
            "key_principles": [],
            "recommended_action": "Check your technical setup and try again."
        }
    
    try:
        response: RunResponse = buffett_agent.run(question)
        if response and response.content and isinstance(response.content, InvestmentAdvice):
            return {
                "response": response.content.response,
                "key_principles": response.content.key_principles,
                "recommended_action": response.content.recommended_action
            }
        else:
            return {
                "response": "I'm having trouble organizing my thoughts right now. Could you try asking your question again?",
                "key_principles": [],
                "recommended_action": "Please rephrase your question."
            }
    except Exception as e:
        return {
            "response": f"I'm experiencing some technical difficulties: {e}",
            "key_principles": [],
            "recommended_action": "Please try your question again."
        }

def chat_with_soros(question):
    """Get George Soros's response to a question."""
    if not AGNO_READY or not soros_agent:
        return {
            "response": "I regret that I'm not available for discussion at the moment due to technical constraints. Please verify your OpenAI API configuration.",
            "key_principles": [],
            "recommended_action": "Resolve technical issues and reconnect."
        }
    
    try:
        response: RunResponse = soros_agent.run(question)
        if response and response.content and isinstance(response.content, InvestmentAdvice):
            return {
                "response": response.content.response,
                "key_principles": response.content.key_principles,
                "recommended_action": response.content.recommended_action
            }
        else:
            return {
                "response": "My thoughts seem scattered at the moment. Perhaps you could reformulate your inquiry?",
                "key_principles": [],
                "recommended_action": "Please ask your question differently."
            }
    except Exception as e:
        return {
            "response": f"I'm encountering technical impediments: {e}",
            "key_principles": [],
            "recommended_action": "Please retry your inquiry."
        }

def display_response(name, color, response_data):
    """Display a formatted response from an advisor."""
    print(f"\n{color + Style.BRIGHT}{name}:{Style.RESET_ALL}")
    print(f"{color}{'─' * 60}{Style.RESET_ALL}")
    print(f"{response_data['response']}")
    
    if response_data['key_principles']:
        print(f"\n{color + Style.BRIGHT}Key Principles:{Style.RESET_ALL}")
        for principle in response_data['key_principles']:
            print(f"{color}• {principle}{Style.RESET_ALL}")
    
    if response_data['recommended_action'] and response_data['recommended_action'].strip():
        print(f"\n{color + Style.BRIGHT}Recommended Action:{Style.RESET_ALL}")
        print(f"{color}{response_data['recommended_action']}{Style.RESET_ALL}")
    
    print(f"{color}{'─' * 60}{Style.RESET_ALL}")

def main():
    """Main chat application loop."""
    if not initialize_agno_agents():
        print("Failed to initialize chat agents. Please check your setup and try again.")
        return
    
    clear_screen()
    print_header()
    
    print("\nWelcome to Investment Strategy Chat!")
    print("Get personalized investment advice and insights from legendary investors.")
    print("Ask questions, share scenarios, or discuss market philosophies.")
    
    chat_history = []
    
    while True:
        print_menu()
        choice = input(f"\n{Fore.CYAN}Enter your choice (1-5): {Style.RESET_ALL}")
        
        if choice == '5':
            print(f"\n{Fore.YELLOW}Thank you for chatting with the investment legends!{Style.RESET_ALL}")
            print("Remember: Past performance does not guarantee future results. Always do your own research!")
            break
        
        elif choice == '4':
            clear_screen()
            print_header()
            print_topics()
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            continue
        
        elif choice in ['1', '2', '3']:
            print(f"\n{Fore.YELLOW}Enter your investment question or topic:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}(Type 'back' to return to menu, 'history' to see chat history){Style.RESET_ALL}")
            
            question = input(f"\n{Fore.CYAN}Your question: {Style.RESET_ALL}")
            
            if question.lower() == 'back':
                continue
            elif question.lower() == 'history':
                if chat_history:
                    print(f"\n{Fore.YELLOW + Style.BRIGHT}Chat History:{Style.RESET_ALL}")
                    for i, (q, advisor) in enumerate(chat_history[-5:], 1):  # Show last 5
                        print(f"{i}. {Fore.CYAN}{q}{Style.RESET_ALL} (asked to {advisor})")
                else:
                    print(f"\n{Fore.YELLOW}No chat history yet.{Style.RESET_ALL}")
                input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
                continue
            elif not question.strip():
                print(f"{Fore.RED}Please enter a valid question.{Style.RESET_ALL}")
                continue
            
            # Add timestamp and context to question for better responses
            enhanced_question = f"Question asked on {datetime.now().strftime('%B %d, %Y')}: {question}"
            
            print(f"\n{Fore.YELLOW}Getting response(s)...{Style.RESET_ALL}")
            
            if choice == '1':  # Warren Buffett
                response = chat_with_buffett(enhanced_question)
                display_response("Warren Buffett", Fore.GREEN, response)
                chat_history.append((question, "Warren Buffett"))
                
            elif choice == '2':  # George Soros
                response = chat_with_soros(enhanced_question)
                display_response("George Soros", Fore.MAGENTA, response)
                chat_history.append((question, "George Soros"))
                
            elif choice == '3':  # Both
                print(f"\n{Fore.YELLOW}Getting responses from both advisors...{Style.RESET_ALL}")
                
                buffett_response = chat_with_buffett(enhanced_question)
                display_response("Warren Buffett", Fore.GREEN, buffett_response)
                
                soros_response = chat_with_soros(enhanced_question)
                display_response("George Soros", Fore.MAGENTA, soros_response)
                
                chat_history.append((question, "Both"))
            
            # Ask if user wants to follow up
            follow_up = input(f"\n{Fore.CYAN}Would you like to ask a follow-up question? (y/n): {Style.RESET_ALL}")
            if follow_up.lower() != 'y':
                continue
        
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()