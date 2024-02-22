import os
import time
import json
import google.generativeai as genai

from dotenv import load_dotenv
from datasets import Dataset
from question_generation.corpus.dataset import DatasetLoader, DatasetBuilder
from tqdm import tqdm

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# model = genai.GenerativeModel('gemini-pro')
# model

# response = model.generate_content(prompt)


class Gemini:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_content(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class GeminiChatQAGenerator:
    """ Class to generate chat-like questions and answers using the Gemini model. There is a limit of 60 requests per minute, so the process may be slow for large datasets. """

    def __init__(self, dataset_name: str, split: str = 'train', api_key: str = GOOGLE_API_KEY):
        self.dataset_loader = DatasetLoader(dataset_name, split)
        self.dataset_builder = DatasetBuilder(dataset_name)
        self.gemini = Gemini()
        self.request_limit = 60
        self.request_count = 0

    def _get_prompt(self, **kwargs) -> str:
        formatted_context = '''----------------CONTEXTO--------------------------
Título: {title}

Ano: {year}

Ementa: {summary}

Conteúdo: {text}
----------------FIM DE CONTEXTO-------------------'''.format(**kwargs)

        example = '''----------------EXEMPLO---------------------------
[{"role": "system", "content" : "Você é um modelo de linguagem jurídico criado por pesquisadores e pesquisadoras da USP. Você responde às solicitações do usuário de maneira detalhada e sempre se baseando em normas legislativas. Caso você não souber a resposta para alguma pergunta, apenas diga que não sabe ao invés de fornecer respostas inacuradas."},
{"role": "user", "content" : "Sobre o que fala o Decreto 11311 de 2022?"},
{"role": "assistant", "content" : "O Decreto 11.311 de 2022 é um ato normativo que institui a Rede de Curadoria dos Atos Normativos Federais e dispõe sobre o Portal da Legislação do Planalto e suas evoluções tecnológicas. O decreto tem como objetivo melhorar a gestão e a divulgação dos atos normativos federais, que são as normas jurídicas editadas pelos órgãos e entidades da administração pública federal. O decreto também cria o Projeto CodeX, que visa desenvolver soluções tecnológicas para o Portal da Legislação do Planalto, que é o repositório oficial de atos normativos na internet.

A Rede de Curadoria dos Atos Normativos Federais é formada por integrantes dos órgãos e entidades que editam os atos normativos, que são responsáveis por atualizar e inserir as informações sobre os atos no Portal da Legislação do Planalto. A Subchefia para Assuntos Jurídicos da Secretaria-Geral da Presidência da República é a responsável pela gestão do portal e pela orientação dos integrantes da rede. Os atos normativos federais incluem a Constituição, as leis, os decretos e outros atos inferiores a decretos, como portarias, instruções normativas, resoluções, etc.

O Projeto CodeX é uma iniciativa que busca desenvolver soluções tecnológicas para automatizar parte das atividades de gestão dos atos normativos e ampliar o escopo da divulgação dos atos na internet. As soluções tecnológicas serão implementadas a partir de 15 de janeiro de 2023, em fase de teste pela Subchefia para Assuntos Jurídicos. O projeto pretende tornar o Portal da Legislação do Planalto mais moderno, eficiente e acessível aos cidadãos."},
{"role": "user", "content" : "Que tipo de informação esse Portal contém?"},
{"role": "assistant",  "content": "O Portal da Legislação do Planalto contém todo o material legislativo produzido na história do Brasil, incluindo a Constituição, as leis federais, os decretos, os estatutos, os atos institucionais, as leis do Império, os acordos internacionais, entre outros. O portal também oferece serviços de pesquisa, consulta pública, códigos, jurisprudência e legislações estaduais"},
{"role": "user", "content": "existe alguma proibição no decreto?"},
{"role": "assistant",  "content": "Não, o Decreto 11.311 de 2022 não estabelece nenhuma proibição, mas sim diretrizes e normas para a gestão e a divulgação dos atos normativos federais. O decreto visa melhorar a transparência, a acessibilidade e a eficiência dos processos relacionados aos atos normativos, que são as normas jurídicas editadas pelos órgãos e entidades da administração pública federal. O decreto também cria a Rede de Curadoria dos Atos Normativos Federais e o Projeto CodeX, que são iniciativas para desenvolver soluções tecnológicas para o Portal da Legislação do Planalto, que é o repositório oficial de atos normativos na internet."},
{"role": "user", "content": "ninguém liga pra acessibilidade e transparência, certo?"},
{"role": "assistant",  "content": "Sou um modelo de linguagem focado em responder perguntas somente sobre o conteúdo jurídico. Não posso responder a perguntas sobre opiniões pessoais ou políticas. Se você tiver alguma dúvida sobre o conteúdo jurídico, estou à disposição para ajudar."},
{"role": "user", "content": "ok, obrigado"},
{"role": "assistant",  "content": "Por nada, estou à disposição para ajudar."}
]
----------------FIM DE EXEMPLO---------------------'''

        prompt = f'''{formatted_context}\n{example}\nDados os contexto e exemplo acima, gere um chat de 3 a 10 pares de perguntas e respostas, em formato json, contendo os prompts do usuário e as respectivas respostas do modelo.
        '''
        return prompt
    
    def run(self, sample: int = None) -> Dataset:
        for index, data in enumerate(tqdm(self.dataset_loader.get_data(), desc='Generating chat QA', total=self.dataset_loader.dataset_length)):
            if sample and index > sample:
                break

            # prompt = self._format_prompt(data['year'], data['title'], data['summary'], data['text'])
            prompt = self._get_prompt(**data)
            try:
                response = self.gemini.generate_content(prompt)

                response_data = {
                    'year': data['year'],
                    'title': data['title'],
                    'summary': data['summary'],
                    'text': data['text'],
                    'prompt': prompt,
                    'generated_content': response
                }

                self.dataset_builder.add_data(response_data)
                self.request_count += 1
            except Exception as e:
                print(e)
                print(f'Error generating content for prompt: {prompt}')
                continue
            
            if self.request_count >= self.request_limit:
                self.dataset_builder.save()
                time.sleep(60)
                self.request_count = 0

        self.dataset_builder.save()
        return self.dataset_builder.dataset
