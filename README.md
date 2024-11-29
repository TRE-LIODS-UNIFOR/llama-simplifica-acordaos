# LLaMa - Simplifica Acórdãos

Resume e simplifica a linguagem de um acórdão do TRE, usando principalmente o LLaMa 3.1:8b.

## Como usar

Clone o repositório

```bash
git clone https://github.com/TRE-LIODS-UNIFOR/llama-simplifica-acordaos.git
cd llama-simplifica-acordaos
```

Crie e ative um ambiente virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Instale as dependências

```bash
cd src
pip install -r requirements.txt
```

Inicie o servidor para ter acesso aos endpoints:

```bash
python3 api.py
```

Envie um acórdão para simplificar:

```bash
curl --location 'http://localhost:5000/simplify' \
--form 'doc=@"[CAMINHO DO DOCUMENTO]"' \
--form 'sections="0"' \  # Página de início do Extrato
--form 'sections="2"' \  # Página de início do Relatório
--form 'sections="5"' \  # Página de início do Voto
--form 'sections="10"' \  # Página de início da Decisão
--form 'sections="12"' \  # Página de fim da Decisão
```

Agora, é só esperar!

## Formato da resposta

A resposta é do mimetype **text/plain**, e contém a versão simplificada do acórdão.
