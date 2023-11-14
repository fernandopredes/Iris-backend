<p align="center">
  <img src="./images/ship-stock-sentry.png" alt="Logo do Ship Stock Sentry">
</p>

# Ship Stock Sentry 🚢

O Ship Stock Sentry é uma aplicação que permite aos usuários monitorar os consumíveis de um navio de perfuração. Com ele você pode realizar um cadastro/login, visualizar todos os relatórios de consumíveis de bordo (ROBs), criar novos ROBs, editar os existentes e excluir ROBs. O back-end da aplicação foi construído usando Flask.

O front-end da aplicação pode ser encontrado no seguinte repositório: [https://github.com/fernandopredes/ShipStockSentry-front](https://github.com/fernandopredes/ShipStockSentry-front)

## Funcionalidades 🚀

- Visualizar todos os ROBs a bordo do navio de perfuração, incluindo informações sobre o tipo de consumível e quantidade disponível.
- Visualizar os consumíveis por níveis de tanque e silos, como combustível, água, barita, bentonita e calcário, para ajudar a gerenciar e monitorar o uso.
- Criar novos ROBs na lista, fornecendo detalhes sobre o tipo de consumível e quantidade disponível.
- Editar ROBs existentes para atualizar as quantidades disponíveis oem caso de erro no preenchimento.
- Excluir ROBs não são mais necessários.

## Instalação 🧰

Requer o [Python 3](https://www.python.org/downloads/) instalado para rodar.

Para usar o Ship Stock Sentry localmente, siga estes passos:

1. Clone o repositório para a sua máquina.
2. Crie um ambiente virtual executando `python3 -m venv venv` no terminal.
3. Ative o ambiente virtual executando `source venv/bin/activate`.
4. Instale as dependências necessárias executando `pip install -r requirements.txt`.
5. Inicie o servidor back-end executando `flask run` no terminal. Isso iniciará o servidor Flask em `http://localhost:5000`.
6. Acesse a documentação da API navegando para `http://localhost:5000/swagger-ui` em seu navegador da web.

## Uso

Antes de poder usar o aplicativo Ship Stock Sentry localmente, tanto o back-end quanto o front-end do aplicativo devem estar em execução. Aqui estão os passos para usar o aplicativo:

1. Inicie o servidor back-end executando `flask run` no terminal. Isso iniciará o servidor Flask em um localhost. Um exemplo da aplicação inicada na porta 5000 pode ser acessado em: `http://localhost:5000`.
2. Em uma janela de terminal separada, navegue até o diretório front-end e siga as instruções no arquivo readme.md do repositório front-end para iniciar o aplicativo front-end.
3. Registre-se e faça login no aplicativo. A partir daí, você pode criar novos ROBs, editar os existentes e excluir ROBs conforme necessário. Você também pode visualizar os consumíveis por níveis de tanque para ajudar a gerenciar e monitorar o uso.

## Contribuindo

Se você gostaria de contribuir com o Ship Stock Sentry, abra um pull request ou uma issue no repositório do GitHub.
