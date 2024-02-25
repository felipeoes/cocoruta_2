import os
import time
import random
import google.generativeai as genai
import requests

from dotenv import load_dotenv
from question_generation.corpus.dataset import DatasetLoader, DatasetBuilder
# from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Timer
from tqdm import tqdm
from multiprocessing import Process, Queue, Event, cpu_count, Value
from queue import Empty

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_0")
genai.configure(api_key=GOOGLE_API_KEY)

# class Gemini(Thread):


class Gemini(Process):
    """ Class to generate content using the Gemini model """

    def __init__(self, api_key: str, queue: Queue, response_queue: Queue, sleep_event: Event, shared_request_count: Value, verbose: bool = False):  # type: ignore
        super().__init__()
        self.api_key = api_key
        self.queue = queue
        self.response_queue = response_queue
        self.sleep_event = sleep_event
        self.shared_request_count = shared_request_count
        self.verbose = verbose
        self.model = genai.GenerativeModel('gemini-pro')
        self.stop_event = False
        self.sleep_time = 60
        self.max_tries = 3
        self.current_tries = 0  # reset the current_tries counter whenever a response is received

    def _process_item_rest_api(self, prompt: str) -> str:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}'
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt}]}],
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ],
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        if result.get('promptFeedback') and result['promptFeedback'].get('blockReason'):
            print(
                f'Prompt feedback: {result["promptFeedback"]["blockReason"]}')
            raise Exception(
                f'Prompt feedback block reason: {result["promptFeedback"]}')

        finish_reason = result['candidates'][0]['finishReason']
        if finish_reason != 'STOP':
            print(f'Finish reason: {finish_reason}')

        return result['candidates'][0]['content']['parts'][0]['text']

    def _process_item(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        if response.candidates[0].finish_reason.value > 1:
            print(
                f"Response finish reason: {response.candidates[0].finish_reason} | Value: {response.candidates[0].finish_reason.value}")

        return response.text

    def run(self):
        while True:
            if self.sleep_event.is_set():
                print(
                    f'Worker {self.name} is sleeping for {self.sleep_time} seconds')
                time.sleep(self.sleep_time)

            if self.stop_event and self.queue.empty():  # only stop if the queue is empty and the stop_event is set
                break

            # wait a random time between 3 and 6 seconds to avoid reaching the request limit so quickly
            time.sleep(random.randint(3, 10))

            try:
                item = self.queue.get(timeout=1)

                # response = self._process_item(item['prompt'])
                response = self._process_item_rest_api(
                    item['prompt'])  # testing the rest api
                self.response_queue.put({
                    'response': response,
                    'prompt': item['prompt'],
                    'data': item['data']
                })

                if self.verbose:
                    print(
                        f'Worker {self.name} processed item {item["data"]["title"]}')

                self.current_tries = 0

            except (Empty, TimeoutError) as e:
                print(e)
                self.current_tries += 1
                print(
                    f'Empty queue, retrying {self.current_tries} of {self.max_tries}')
                time.sleep(self.sleep_time)

                if self.current_tries >= self.max_tries:
                    self.stop_event = True

            except Exception as e:
                print(e)
                print('Error generating content, adding to queue again')
                self.queue.put(item)

            finally:
                with self.shared_request_count.get_lock():
                    self.shared_request_count.value += 1

                # wait a random time between 3 and 6 seconds to avoid reaching the request limit so quickly
                time.sleep(random.randint(3, 10))


class GeminiChatQAGenerator:
    """ Class to generate chat-like questions and answers using the Gemini model. There is a limit of 60 requests per minute, so the process may be slow for large datasets. """

    def __init__(self, dataset_name: str, split: str = 'train', num_api_keys: int = 1,
                 #  num_workers: int = min(cpu_count() // 2, 10),
                 verbose: bool = False):
        self.dataset_loader = DatasetLoader(dataset_name, split)
        self.dataset_builder = DatasetBuilder(dataset_name)
        # at most 10 workers to avoid reaching the request limit so quickly
        self.num_api_keys = num_api_keys
        # set 20 workers for each api key
        self.num_workers = 10 * num_api_keys
        self.verbose = verbose
        self.gemini_workers = []
        # setting 40 instead of 60 because the api is not working properly (it is returning 429 error even with less than 60 requests)
        # multiplied the num_api_keys with the request limit since the request limit is per api key
        self.request_limit = 60 * num_api_keys
        self.prompt_queue = Queue()
        self.response_queue = Queue()
        self.sleep_event = Event()
        self.shared_request_count = Value('i', 0)
        self._initialize_gemini_workers()

    def _initialize_gemini_workers(self):
        api_keys = [os.getenv(f"GOOGLE_API_KEY_{i}")
                    for i in range(self.num_api_keys)]

        # distribution of workers among the api keys
        for i, api_key in enumerate(api_keys):
            for _ in range(self.num_workers // self.num_api_keys):
                worker = Gemini(api_key,
                                self.prompt_queue,
                                self.response_queue, self.sleep_event, self.shared_request_count, self.verbose)
                worker.start()
                self.gemini_workers.append(worker)

        # check how many workers were created for each api key
        keys = {api_key: 0 for api_key in api_keys}
        for worker in self.gemini_workers:
            keys[worker.api_key] += 1

        print(keys)

        # for _ in range(self.num_workers):
        #     worker = Gemini(GOOGLE_API_KEY,
        #                     self.prompt_queue,
        #                     self.response_queue, self.sleep_event, self.shared_request_count, self.verbose)
        #     worker.start()
        #     self.gemini_workers.append(worker)

    def _get_prompt(self, **kwargs) -> str:
        formatted_context = '''----------------{title}--------------------------
Título: {title}\nAno: {year}\nEmenta: {summary}\nConteúdo: {text}
----------------------------------'''.format(**kwargs)

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
{"role": "assistant",  "content": "Sou um modelo de linguagem focado em responder perguntas somente sobre o conteúdo jurídico. Não posso responder a perguntas que envolvam opiniões pessoais, políticas ou de cunho discriminatório. Se você tiver alguma dúvida sobre o contexto jurídico brasileiro, estou à disposição para ajudar."},
{"role": "user", "content": "ok, obrigado"},
{"role": "assistant",  "content": "Por nada, estou à sua disposição."}
]
----------------FIM DE EXEMPLO---------------------'''

        prompt = f'''{formatted_context}\n{example}\nDada a norma e o exemplo acima, gere um chat de 3 a 5 pares de perguntas e respostas (detalhadas e completas), em formato json, contendo os prompts do usuário e as respectivas respostas do modelo em relação ao {kwargs.get('title')}. Inclua o título da norma ({kwargs.get('title')}) nas respostas quando for pertinente.
        '''
        return prompt

    def _reset_request_count(self):
        """ Reset the request count """
        if self.shared_request_count.value >= self.request_limit:
            print('Clearing sleep event after 60 seconds')
            Timer(60, self.sleep_event.clear).start()

            self.sleep_event.set()

            self.dataset_builder.save()
            self.shared_request_count.value = 0

    def run(self, start: int = 0, end: int = None):
        """ Run the chat QA generation process """

        # total_data = self.dataset_loader.dataset_length - \
        #     self.dataset_builder.dataset_length if not end else end
        # start = start if start > self.dataset_builder.dataset_length else self.dataset_builder.dataset_length - 1

        # only process remaining data (which is not in the dataset builder yet)
        remaining_data = []
        dataset_titles = self.dataset_builder.get_titles().to_list()

        for data in self.dataset_loader.get_rows():
            if data['title'] not in dataset_titles:
                remaining_data.append(data)

        total_data = len(remaining_data)

        for index, data in enumerate(tqdm(remaining_data, desc='Generating chat QA', total=total_data)):
            if index < start:
                continue

            if end and index > end:
                break

            prompt = self._get_prompt(**data)
            self.prompt_queue.put({
                'data': data,
                'prompt': prompt
            })

        print(f'Prompt queue size: {self.prompt_queue.qsize()}')

        pbar = tqdm(desc='Processing responses',
                    total=total_data)
        while any([worker.is_alive() for worker in self.gemini_workers]):
            result = self.response_queue.get()

            if not result:
                continue

            response_data = {
                'year': result['data']['year'],
                'title': result['data']['title'],
                'summary': result['data']['summary'],
                'text': result['data']['text'],
                'prompt': result['prompt'],
                'generated_content': result['response']
            }

            self.dataset_builder.add_data(response_data)

            pbar.update(1)
            self._reset_request_count()

        for worker in self.gemini_workers:
            worker.stop_event = True

        self.dataset_builder.save()
        return self.dataset_builder.get_hf_dataset()
