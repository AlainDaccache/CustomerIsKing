import gradio as gr
import os


def launch_app():

    # TODO abstract away

    s = """
    Welcome to Dunya, the intelligent Chatbot for our Fictional Comany! 
    I am here to assist and guide you through various tasks. Currently, I excel in:
      • Retrieving and synthesizing information from our extensive knowledge base, covering processes, regulations, and more. Feel free to ask questions such as your rights and obligations for an employee or contractor.
      
    In the near future, I will also be capable of extracting data from our database, enhancing my capabilities even further.
    """

    LLM_HOST = os.getenv("LLM_HOST")
    LLM_PORT = os.getenv("LLM_PORT")

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            [[None, s]],
        )
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        import requests

        def respond(question, chat_history):
            print("Question:", question)
            bot_message = requests.get(
                f"http://{LLM_HOST}:{LLM_PORT}/answer/?question={question}"
            ).json()["response"]
            chat_history.append((question, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
        ssl_verify=False,
    )


if __name__ == "__main__":
    launch_app()
