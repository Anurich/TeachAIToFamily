import gradio as gr
import uuid
from datetime import datetime
import json
from agentic import get_workflow

# Import your workflow function
# from your_module import get_workflow  # Replace with your actual import

# Initialize the workflow
print("🚀 Initializing LangGraph workflow...")
try:
    workflow = get_workflow()
    print("✅ Workflow initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing workflow: {e}")
    workflow = None

# Global variable to store conversation state
conversation_state = {}

def chat_with_workflow(message, history, thread_id):
    """
    Main chat function that integrates with your LangGraph workflow
    """
    global conversation_state
    
    if not workflow:
        return history + [["Error: Workflow not initialized", None]], thread_id
    
    if not message.strip():
        return history, thread_id
    
    # Generate thread ID if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"🆕 New conversation started with thread ID: {thread_id}")
    
    try:
        print(f"📨 Processing message: {message}")
        print(f"🧵 Thread ID: {thread_id}")
        
        # Invoke your workflow with the exact format you specified
        result = workflow.invoke(
            {
                "messages": [("user", message)]
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the bot response from the workflow result
        if "messages" in result and len(result["messages"]) > 0:
            # Get the last message which should be the bot's response
            last_message = result["messages"][-1]
            print(last_message)
            
            # Handle different message formats
            if isinstance(last_message, tuple) and len(last_message) >= 2:
                bot_response = last_message[1]  # (role, content) format
            elif hasattr(last_message, 'content'):
                bot_response = last_message.content  # Message object with content attribute
            elif isinstance(last_message, str):
                bot_response = last_message
            else:
                bot_response = str(last_message)
        else:
            bot_response = "I apologize, but I couldn't generate a response. Please try again."
        
        print(f"🤖 Bot response: {bot_response}")
        
        # Update conversation history
        new_history = history + [[message, bot_response]]
        
        # Store conversation state
        conversation_state[thread_id] = {
            "history": new_history,
            "last_updated": datetime.now().isoformat()
        }
        
        return new_history, thread_id
        
    except Exception as e:
        error_message = f"❌ Error processing request: {str(e)}"
        print(error_message)
        return history + [[message, error_message]], thread_id

def clear_conversation():
    """Clear the current conversation"""
    return [], ""

def export_conversation(history, thread_id):
    """Export conversation history as JSON"""
    if not history:
        return None
    
    export_data = {
        "thread_id": thread_id,
        "export_time": datetime.now().isoformat(),
        "conversation": [
            {
                "user": msg[0] if msg[0] else "",
                "assistant": msg[1] if msg[1] else "",
                "timestamp": datetime.now().isoformat()
            }
            for msg in history if msg[0] is not None
        ]
    }
    
    filename = f"chat_export_{thread_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return filename

def get_workflow_info():
    """Get information about the current workflow"""
    if workflow:
        try:
            # Try to get workflow graph information
            info = {
                "status": "✅ Initialized",
                "type": str(type(workflow)),
                "thread_support": "✅ Enabled",
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"✅ Workflow initialized but info unavailable: {str(e)}"
    else:
        return "❌ Workflow not initialized"

# Your existing workflow function (replace this with your actual implementation)
def get_workflow():
    """
    Replace this with your actual get_workflow function
    This is just a placeholder - uncomment and use your actual code below
    """
    
    # Uncomment and replace with your actual workflow code:
    """
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import InMemorySaver
    
    graph = StateGraph(state_schema=State)
    graph.add_node("boss", node_1)
    graph.add_node("node_2", agent_2)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_answer", final_answer)
    graph.set_entry_point("boss")
    graph.add_conditional_edges(
        "boss", conditional_check, 
        {
            "call_next": "node_2",
            "__end__": END
        }
    )
    graph.add_conditional_edges("node_2", should_continue, {
        "continue": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "final_answer")
    graph.add_edge("final_answer", END)
    
    workflow = graph.compile(checkpointer=InMemorySaver())
    
    # Optional: Save graph visualization
    try:
        import os 
        png_graph_bytes = workflow.get_graph().draw_mermaid_png()
        output_file_path = "my_langgraph.png"
        with open(output_file_path, "wb") as f:
            f.write(png_graph_bytes)
        print(f"Graph saved as '{output_file_path}' in {os.getcwd()}")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")
    
    return workflow
    """
    
    # Mock workflow for demonstration - remove this when using your actual code
    class MockWorkflow:
        def invoke(self, data, config=None):
            user_msg = data["messages"][0][1]
            thread_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
            
            # Simulate different types of responses based on message content
            if "error" in user_msg.lower():
                response = "This is a simulated error response for testing."
            elif "hello" in user_msg.lower() or "hi" in user_msg.lower():
                response = f"Hello! I'm your LangGraph assistant. Thread ID: {thread_id[:8]}..."
            elif "workflow" in user_msg.lower():
                response = "I'm powered by a LangGraph workflow with multiple nodes including boss, agent_2, tool_node, and final_answer nodes."
            else:
                response = f"I received your message: '{user_msg}'. This is a response from the LangGraph workflow simulation."
            
            return {
                "messages": [
                    ("user", user_msg),
                    ("assistant", response)
                ]
            }
    
    return MockWorkflow()

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}

.chat-message {
    padding: 10px;
    margin: 5px 0;
    border-radius: 10px;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}

.bot-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}

.header-text {
    text-align: center;
    color: #1976d2;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

.status-text {
    padding: 10px;
    background-color: #e8f5e8;
    border-radius: 5px;
    margin: 10px 0;
    font-family: monospace;
    font-size: 12px;
}
"""

# Create the Gradio interface
def create_gradio_interface():
    with gr.Blocks(
        title="🤖 LangGraph AI Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML("<h1 style='text-align: center; color: #1976d2;'>🤖 LangGraph AI Assistant</h1>")
        gr.HTML("<p style='text-align: center; color: #666;'>Powered by your custom LangGraph workflow</p>")
        
        # State variables
        thread_id_state = gr.State("")
        
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        label="Conversation",
                        placeholder="Start a conversation with your LangGraph assistant...",
                        bubble_full_width=False,
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message here... (Press Enter to send)",
                            label="Message",
                            lines=2,
                            max_lines=5,
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    # Control panel
                    gr.HTML("<h3>🎛️ Controls</h3>")
                    
                    thread_display = gr.Textbox(
                        label="Current Thread ID",
                        placeholder="Auto-generated on first message",
                        interactive=False,
                        lines=2
                    )
                    
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    new_thread_btn = gr.Button("🆕 New Thread", variant="secondary")
                    export_btn = gr.Button("📥 Export Chat", variant="secondary")
                    
                    # File output for export
                    export_file = gr.File(label="Exported Chat", visible=False)
        
        with gr.Tab("📊 Workflow Info"):
            gr.HTML("<h3>🔍 Workflow Status</h3>")
            workflow_info = gr.Textbox(
                label="Workflow Information",
                value=get_workflow_info(),
                lines=10,
                interactive=False
            )
            refresh_info_btn = gr.Button("🔄 Refresh Info", variant="secondary")
        
        with gr.Tab("📖 Instructions"):
            gr.Markdown("""
            ## 🚀 How to Use This Interface
            
            ### 💬 Chat Tab
            - Type your message in the text box and press Enter or click Send
            - The conversation maintains context using thread IDs
            - Use "Clear Chat" to start fresh or "New Thread" to begin a new conversation thread
            - Export your conversation as a JSON file using "Export Chat"
            
            ### 🔧 Your LangGraph Workflow
            This interface integrates with your custom LangGraph workflow that includes:
            - **Boss Node**: Initial processing
            - **Agent Node**: Secondary processing  
            - **Tool Node**: Tool execution
            - **Final Answer Node**: Response generation
            
            ### 🧵 Thread Management
            - Each conversation gets a unique thread ID for context preservation
            - Thread IDs are automatically generated or you can start new threads
            - The workflow maintains conversation history using LangGraph's checkpointer
            
            ### ⚙️ Technical Details
            - Built with Gradio for easy deployment
            - Integrates directly with your `get_workflow()` function
            - Supports the exact invoke format: `workflow.invoke({"messages": [("user", message)]}, config={"configurable": {"thread_id": thread_id}})`
            - Handles various response formats from your workflow
            
            ### 🎯 To Use Your Actual Workflow
            1. Replace the mock `get_workflow()` function with your actual implementation
            2. Ensure all your imports and dependencies are available
            3. The interface will automatically use your workflow nodes and logic
            """)
        
        # Event handlers
        def send_message(message, history, thread_id):
            if message.strip():
                new_history, new_thread_id = chat_with_workflow(message, history, thread_id)
                return "", new_history, new_thread_id, new_thread_id
            return message, history, thread_id, thread_id
        
        def clear_chat():
            return [], "", ""
        
        def new_thread():
            return [], "", ""
        
        def export_chat(history, thread_id):
            if history:
                filename = export_conversation(history, thread_id)
                return gr.File(filename, visible=True)
            return gr.File(visible=False)
        
        def refresh_workflow_info():
            return get_workflow_info()
        
        # Connect event handlers
        msg_input.submit(
            send_message,
            inputs=[msg_input, chatbot, thread_id_state],
            outputs=[msg_input, chatbot, thread_id_state, thread_display]
        )
        
        send_btn.click(
            send_message,
            inputs=[msg_input, chatbot, thread_id_state],
            outputs=[msg_input, chatbot, thread_id_state, thread_display]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, thread_id_state, thread_display]
        )
        
        new_thread_btn.click(
            new_thread,
            outputs=[chatbot, thread_id_state, thread_display]
        )
        
        export_btn.click(
            export_chat,
            inputs=[chatbot, thread_id_state],
            outputs=[export_file]
        ).then(
            lambda: gr.File(visible=True),
            outputs=[export_file]
        )
        
        refresh_info_btn.click(
            refresh_workflow_info,
            outputs=[workflow_info]
        )
        
        # Initialize with welcome message
        demo.load(
            lambda: (
                [["", "👋 Welcome! I'm your LangGraph AI assistant. How can I help you today?"]],
                get_workflow_info()
            ),
            outputs=[chatbot, workflow_info]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=True,             # Enable debug mode
        show_error=True,        # Show errors in the interface
        quiet=False             # Show startup logs
    )
    
    # Alternative launch options:
    # demo.launch(share=True)  # Creates a public shareable link
    # demo.launch(auth=("username", "password"))  # Add authentication
    # demo.launch(ssl_verify=False)  # For development with HTTPS issues