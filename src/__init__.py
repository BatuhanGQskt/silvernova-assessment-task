import logging
from src.operations.ask import LLMAsker
from src.operations.search import SearchEngine
from src.operations.embed import EmbedService
from src.operations.extract import MarkdownExtractor
import argparse

class App:
  """ The main class of the application. """

  def __init__(self):
    pass

  def run(self):
    parser = argparse.ArgumentParser(description='Ask questions about the files of a case.')
    
    # Add optional "mode" argument (with values "load-files" and "ask-question" (default))
    parser.add_argument('--mode', choices=['index-files', 'ask-question', 'search', 'get-markdown', 'load-files'], default='ask-question', help='The mode of the application.')

    # Add question argument as required positional argument if mode is "ask-question"
    parser.add_argument('question', nargs='?', type=str, help='The question to ask about the files of a case.')

    args = parser.parse_args()

    if args.mode == 'load-files':
      self.load_files()
    elif args.mode == 'ask-question':
      question = args.question
      if not question or question.isspace():
        parser.error('The question argument is required in "ask-question" mode.')
      self.ask_question(question)
    elif args.mode == 'search':
      question = args.question
      if not question or question.isspace():
        parser.error('The query argument is required in "search" mode.')
      self.search(question)
    elif args.mode == 'get-markdown':
      self.get_markdown()

  def load_files(self):
    # ToDo: Load the files and index them in a db of your choosing for the rag
    
    operator = EmbedService()
    operator.embed('This is a test')

    # ...

  def search(self, query):
    # ToDo: Search the indexed files for results matching your query
    pass

  def get_markdown(self):
    # ToDo
    pass

  def ask_question(self, question):
    logging.info(f'Asking question: {question}')

    operator = LLMAsker()

    response = operator.ask(question)

    print(response)