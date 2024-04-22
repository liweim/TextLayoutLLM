try:
    from .llm import load_llm, limit_token_length
except Exception as e:
    print(e)
    from llm import load_llm, limit_token_length


class DocQA:
    def __init__(self, model_type, use_cache=False, instruction='', show_log=True, language='en', **kwargs):
        self.llm, self.tokenizer, self.max_length = load_llm(model_type, use_cache=use_cache, **kwargs)
        self.instruction = instruction
        self.show_log = show_log

        if self.instruction == '':
            if language == 'zh':
                self.prompt_template = """基于以下内容:
{context}

回答问题:
{question}"""
            else:
                self.prompt_template = """Given the context:
{context}

Answer the question:
{question}"""
        else:
            if model_type.startswith('llama2') and '[INST]' not in instruction:
                tmp = 'Question: {question}\nAnswer: '
                instruction = "<s>[INST] <<SYS>>\n" + instruction.replace(tmp, '') + "<</SYS>>\n\n{question} [/INST]"
                print("llama2's instruction is different: \n"+instruction)
            self.prompt_template = instruction


    def __call__(self, context, question):
        context, exceed_limit = limit_token_length(self.tokenizer, self.max_length, self.prompt_template, context,
                                                   question)
        question_prompt = self.prompt_template.format(context=context, question=question)
        answer = self.llm(question_prompt)
        if self.show_log:
            print('*' * 100)
            print(f'context:\n{context}')
            print(f'question: {question}')
            print(f'answer: {answer}')
        return answer, exceed_limit
